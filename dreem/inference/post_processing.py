"""Helper functions for post-processing association matrix pre-tracking."""

import torch
from dreem.inference.boxes import Boxes


def weight_decay_time(
    asso_output: torch.Tensor,
    decay_time: float = 0,
    reid_features: torch.Tensor | None = None,
    T: int | None = None,
    k: int | None = None,
) -> torch.Tensor:
    """Weight association matrix by time.

    Weighs matrix by number of frames the ith object is from the jth object
    in the association matrix.

    Args:
        asso_output: the association matrix to be reweighted
        decay_time: the scale to weight the asso_output by
        reid_features: The n x d matrix of feature vectors for each object
        T: The length of the window
        k: an integer for the query frame within the window of instances
    Returns: The N_t x N association matrix weighted by decay time
    """
    if decay_time is not None and decay_time > 0:
        assert (
            reid_features is not None and T is not None and k is not None
        ), "Need reid_features to weight traj_score by `decay_time`!"
        N_t = asso_output.shape[0]
        dts = torch.cat(
            [
                x.new_full((N_t,), T - t - 2)
                for t, x in enumerate(reid_features)
                if t != k
            ],
            dim=0,
        ).cpu()  # Np
        # asso_output = asso_output.to(self.device) * (self.decay_time ** dts[None, :])
        asso_output = asso_output * (decay_time ** dts[:, None])
    return asso_output


def _pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute the intersection area between __all__ N x M pairs of boxes.

    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, :, 2:], boxes2[:, :, 2:]) - torch.max(
        boxes1[:, None, :, :2], boxes2[:, :, :2]
    )  # [N,M,n_anchors,     2]
    width_height.clamp_(min=0)  # [N,M, n_anchors, 2]

    intersection = width_height.prod(dim=3)  # [N,M, n_anchors]

    return intersection


def _pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute intersection over union between all N x M pairs of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]

    inter = _pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter >= 0,
        inter / (area1[:, None, :] + area2 - inter),
        torch.nan,
    )
    return iou.nanmean(dim=-1)


def weight_iou(
    asso_output: torch.Tensor, method: str | None = None, last_ious: torch.Tensor = None
) -> torch.Tensor:
    """Weight the association matrix by the IOU between object bboxes across frames.

    Args:
        asso_output: An N_t x N association matrix
        method: string indicating whether to use a max weighting or multiplicative weighting
                Max weighting: take `max(traj_score, iou)`
                multiplicative weighting: `iou*weight + traj_score`
        last_ious: torch Tensor containing the ious between current and previous frames

    Returns:
        An N_t x N association matrix weighted by the IOU
    """
    if method is not None and method != "":
        assert last_ious is not None, "Need `last_ious` to weight traj_score by `IOU`"
        if method.lower() == "mult":
            weights = torch.abs(last_ious - asso_output)
            weighted_iou = weights * last_ious
            weighted_iou = torch.nan_to_num(weighted_iou, 0)
            asso_output = asso_output + weighted_iou
        elif method.lower() == "max":
            asso_output = torch.max(asso_output, last_ious)
        else:
            raise ValueError(
                f"`method` must be one of ['mult' or 'max'] got '{method.lower()}'"
            )
    return asso_output


def filter_max_center_dist(
    asso_output: torch.Tensor,
    max_center_dist: float = 0,
    id_inds: torch.Tensor | None = None,
    curr_frame_boxes: torch.Tensor | None = None,
    prev_frame_boxes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Filter trajectory score by distances between objects across frames.

    Args:
        asso_output: An N_t x N association matrix
        max_center_dist: The euclidean distance threshold between bboxes
        id_inds: track ids
        curr_frame_boxes: the raw bbox coords of the current frame instances
        prev_frame_boxes: the raw bbox coords of the previous frame instances

    Returns:
        An N_t x N association matrix
    """
    if max_center_dist is not None and max_center_dist > 0:
        assert (
            curr_frame_boxes is not None
            and prev_frame_boxes is not None
            and id_inds is not None
        ), "Need `curr_frame_boxes`, `prev_frame_boxes`, and `id_ind` to filter by `max_center_dist`"

        k_ct = (curr_frame_boxes[:, :, :2] + curr_frame_boxes[:, :, 2:]) / 2
        # k_s = ((curr_frame_boxes[:, :, 2:] - curr_frame_boxes[:, :, :2]) ** 2).sum(dim=2)  # n_k
        # nonk boxes are only from previous frame rather than entire window
        nonk_ct = (prev_frame_boxes[:, :, :2] + prev_frame_boxes[:, :, 2:]) / 2

        # pairwise euclidean distance in units of pixels
        dist = ((k_ct[:, None, :, :] - nonk_ct[None, :, :, :]) ** 2).sum(dim=-1) ** (
            1 / 2
        )  # n_k x n_nonk
        # norm_dist = dist / (k_s[:, None, :] + 1e-8)

        valid = dist.squeeze() < max_center_dist  # n_k x n_nonk
        # handle case where id_inds and valid is a single value
        # handle this better
        if valid.ndim == 0:
            valid = valid.unsqueeze(0)
        if valid.ndim == 1:
            if id_inds.shape[0] == 1:
                valid_mult = valid.float().unsqueeze(-1)
            else:
                valid_mult = valid.float().unsqueeze(0)
        else:
            valid_mult = valid.float()

        valid_assn = (
            torch.mm(valid_mult, id_inds.to(valid.device)).clamp_(max=1.0).long().bool()
        )  # n_k x M
        asso_output_filtered = asso_output.clone()
        asso_output_filtered[~valid_assn] = 0  # n_k x M
        return asso_output_filtered
    else:
        return asso_output
