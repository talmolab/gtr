"""Module containing logic for going from association -> assignment."""

import torch
import pandas as pd
import logging

from dreem.io import Frame
from dreem.models import model_utils, GlobalTrackingTransformer
from dreem.inference.track_queue import TrackQueue
from dreem.inference import post_processing
from dreem.inference.boxes import Boxes
from scipy.optimize import linear_sum_assignment
from math import inf

logger = logging.getLogger("dreem.inference")


class Tracker:
    """Tracker class used for assignment based on sliding inference from GTR."""

    def __init__(
        self,
        window_size: int = 8,
        use_vis_feats: bool = True,
        overlap_thresh: float = 0.01,
        mult_thresh: bool = True,
        decay_time: float | None = None,
        iou: str | None = None,
        max_center_dist: float | None = None,
        persistent_tracking: bool = False,
        max_gap: int = inf,
        max_tracks: int = inf,
        verbose: bool = False,
    ):
        """Initialize a tracker to run inference.

        Args:
            window_size: the size of the window used during sliding inference.
            use_vis_feats: Whether or not to use visual feature extractor.
            overlap_thresh: the trajectory overlap threshold to be used for assignment.
            mult_thresh: Whether or not to use weight threshold.
            decay_time: weight for `decay_time` postprocessing.
            iou: Either [None, '', "mult" or "max"]
                 Whether to use multiplicative or max iou reweighting.
            max_center_dist: distance threshold for filtering trajectory score matrix.
            persistent_tracking: whether to keep a buffer across chunks or not.
            max_gap: the max number of frames a trajectory can be missing before termination.
            max_tracks: the maximum number of tracks that can be created while tracking.
                We force the tracker to assign instances to a track instead of creating a new track if max_tracks has been reached.
            verbose: Whether or not to turn on debug printing after each operation.
        """
        self.track_queue = TrackQueue(
            window_size=window_size, max_gap=max_gap, verbose=verbose
        )
        self.use_vis_feats = use_vis_feats
        self.overlap_thresh = overlap_thresh
        self.mult_thresh = mult_thresh
        self.decay_time = decay_time
        self.iou = iou
        self.max_center_dist = max_center_dist
        self.persistent_tracking = persistent_tracking
        self.verbose = verbose
        self.max_tracks = max_tracks

    def __call__(
        self, model: GlobalTrackingTransformer, frames: list[Frame]
    ) -> list[Frame]:
        """Wrap around `track` to enable `tracker()` instead of `tracker.track()`.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: list of Frames to run inference on

        Returns:
            List of frames containing association matrix scores and instances populated with pred track ids.
        """
        return self.track(model, frames)

    def __repr__(self) -> str:
        """Get string representation of tracker.

        Returns: the string representation of the tracker
        """
        return (
            "Tracker("
            f"persistent_tracking={self.persistent_tracking}, "
            f"max_tracks={self.max_tracks}, "
            f"use_vis_feats={self.use_vis_feats}, "
            f"overlap_thresh={self.overlap_thresh}, "
            f"mult_thresh={self.mult_thresh}, "
            f"decay_time={self.decay_time}, "
            f"max_center_dist={self.max_center_dist}, "
            f"verbose={self.verbose}, "
            f"queue={self.track_queue}"
        )

    def track(
        self, model: GlobalTrackingTransformer, frames: list[dict]
    ) -> list[Frame]:
        """Run tracker and get predicted trajectories.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: data dict to run inference on

        Returns:
            List of Frames populated with pred track ids and association matrix scores
        """
        # Extract feature representations with pre-trained encoder.

        _ = model.eval()

        for frame in frames:
            if frame.has_instances():
                if not self.use_vis_feats:
                    for instance in frame.instances:
                        instance.features = torch.zeros(1, model.d_model)
                    # frame["features"] = torch.randn(
                    #     num_frame_instances, self.model.d_model
                    # )

                # comment out to turn encoder off

                # Assuming the encoder is already trained or train encoder jointly.
                elif not frame.has_features():
                    with torch.no_grad():
                        crops = frame.get_crops()
                        z = model.visual_encoder(crops)

                        for i, z_i in enumerate(z):
                            frame.instances[i].features = z_i

        # I feel like this chunk is unnecessary:
        # reid_features = torch.cat(
        #     [frame["features"] for frame in instances], dim=0
        # ).unsqueeze(0)

        # asso_preds, pred_boxes, pred_time, embeddings = self.model(
        #     instances, reid_features
        # )
        # get reference and query instances from TrackQueue and calls _run_global_tracker()
        instances_pred = self.sliding_inference(model, frames)

        # e.g. during train/val, don't track across batches so persistent_tracking is switched off
        if not self.persistent_tracking:
            logger.debug(f"Clearing Queue after tracking")
            self.track_queue.end_tracks()

        return instances_pred

    def sliding_inference(
        self, model: GlobalTrackingTransformer, frames: list[Frame]
    ) -> list[Frame]:
        """Perform sliding inference on the input video (instances) with a given window size.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: A list of Frames (See `dreem.io.Frame` for more info).

        Returns:
            frames: A list of Frames populated with pred_track_ids and asso_matrices
        """
        # B: batch size.
        # D: embedding dimension.
        # nc: number of channels.
        # H: height.
        # W: width.

        # frames is untracked clip for inference
        for batch_idx, frame_to_track in enumerate(frames):
            # tracked_frames is a list of reference frames that have been tracked (associated)
            tracked_frames = self.track_queue.collate_tracks(
                device=frame_to_track.frame_id.device
            )
            logger.debug(f"Current number of tracks is {self.track_queue.n_tracks}")

            if (
                self.persistent_tracking and frame_to_track.frame_id == 0
            ):  # check for new video and clear queue

                logger.debug("New Video! Resetting Track Queue.")
                self.track_queue.end_tracks()

            """
            Initialize tracks on first frame where detections appear.
            """
            if len(self.track_queue) == 0:
                if frame_to_track.has_instances():

                    logger.debug(
                        f"Initializing track on clip ind {batch_idx} frame {frame_to_track.frame_id.item()}"
                    )

                    curr_track_id = 0
                    # if track ids exist from another tracking program i.e. sleap, init with those
                    for i, instance in enumerate(frames[batch_idx].instances):
                        instance.pred_track_id = instance.gt_track_id
                        curr_track_id = max(curr_track_id, instance.pred_track_id)
                    # if no track ids, then assign new ones
                    for i, instance in enumerate(frames[batch_idx].instances):
                        if instance.pred_track_id == -1:
                            curr_track_id += 1
                            instance.pred_track_id = curr_track_id

            else:
                if (
                    frame_to_track.has_instances()
                ):  # Check if there are detections. If there are skip and increment gap count
                    # combine the tracked frames with the latest frame; inference pipeline uses latest frame as pred
                    frames_to_track = tracked_frames + [
                        frame_to_track
                    ]  # better var name?

                    query_ind = len(frames_to_track) - 1

                    frame_to_track = self._run_global_tracker(
                        model,
                        frames_to_track,
                        query_ind=query_ind,
                    )

            if frame_to_track.has_instances():
                self.track_queue.add_frame(frame_to_track)
            else:
                self.track_queue.increment_gaps([])
            # update the frame object from the input inference untracked clip
            frames[batch_idx] = frame_to_track
        return frames

    def _run_global_tracker(
        self, model: GlobalTrackingTransformer, frames: list[Frame], query_ind: int
    ) -> Frame:
        """Run global tracker performs the actual tracking.

        Uses Hungarian algorithm to do track assigning.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: A list of Frames containing reid features. See `dreem.io.data_structures` for more info.
            query_ind: An integer for the query frame within the window of instances.

        Returns:
            query_frame: The query frame now populated with the pred_track_ids.
        """
        # *: each item in frames is a frame in the window. So it follows
        #    that each frame in the window has * detected instances.
        # D: embedding dimension.
        # total_instances: number of instances in the window.
        # N_i: number of detected instances in i-th frame of window.
        # instances_per_frame: a list of number of instances in each frame of the window.
        # n_query: number of instances in current/query frame (rightmost frame of the window).
        # n_nonquery: number of instances in the window excluding the current/query frame.
        # window_size: length of window.
        # L: number of decoder blocks.
        # n_traj: number of existing tracks within the window so far.

        # Number of instances in each frame of the window.
        # E.g.: instances_per_frame: [4, 5, 6, 7]; window of length 4 with 4 detected instances in the first frame of the window.

        _ = model.eval()
        # get the last frame in the clip to perform inference on
        query_frame = frames[query_ind]

        query_instances = query_frame.instances
        all_instances = [instance for frame in frames for instance in frame.instances]

        logger.debug(f"Frame {query_frame.frame_id.item()}")

        instances_per_frame = [frame.num_detected for frame in frames]

        total_instances, window_size = sum(instances_per_frame), len(
            instances_per_frame
        )  # Number of instances in window; length of window.

        logger.debug(f"total_instances: {total_instances}")

        overlap_thresh = self.overlap_thresh
        mult_thresh = self.mult_thresh
        n_traj = self.track_queue.n_tracks
        curr_track = self.track_queue.curr_track

        reid_features = torch.cat([frame.get_features() for frame in frames], dim=0)[
            None
        ]  # (1, total_instances, D=512)

        # (L=1, n_query, total_instances)
        with torch.no_grad():
            # GTR knows this is for inference since query_instances is not None
            asso_matrix = model(all_instances, query_instances)

        # GTR output is n_query x n_instances - split this into per-frame to softmax each frame separately
        asso_output = asso_matrix[-1].matrix.split(
            instances_per_frame, dim=1
        )  # (window_size, n_query, N_i)
        asso_output = model_utils.softmax_asso(
            asso_output
        )  # (window_size, n_query, N_i)
        asso_output = torch.cat(asso_output, dim=1).cpu()  # (n_query, total_instances)

        asso_output_df = pd.DataFrame(
            asso_output.clone().numpy(),
            columns=[f"Instance {i}" for i in range(asso_output.shape[-1])],
        )

        asso_output_df.index.name = "Instances"
        asso_output_df.columns.name = "Instances"
        # save the association matrix to the Frame object
        query_frame.add_traj_score("asso_output", asso_output_df)
        query_frame.asso_output = asso_matrix[-1]

        n_query = (
            query_frame.num_detected
        )  # Number of instances in the current/query frame.

        n_nonquery = (
            total_instances - n_query
        )  # Number of instances in the window not including the current/query frame.

        logger.debug(f"n_nonquery: {n_nonquery}")
        logger.debug(f"n_query: {n_query}")

        instance_ids = torch.cat(
            [
                x.get_pred_track_ids()
                for batch_idx, x in enumerate(frames)
                if batch_idx != query_ind
            ],
            dim=0,
        ).view(
            n_nonquery
        )  # (n_nonquery,)

        query_inds = [
            x
            for x in range(
                sum(instances_per_frame[:query_ind]),
                sum(instances_per_frame[: query_ind + 1]),
            )
        ]
        nonquery_inds = [i for i in range(total_instances) if i not in query_inds]

        # instead should we do model(nonquery_instances, query_instances)?
        asso_nonquery = asso_output[:, nonquery_inds]  # (n_query, n_nonquery)

        asso_nonquery_df = pd.DataFrame(
            asso_nonquery.clone().numpy(), columns=nonquery_inds
        )

        asso_nonquery_df.index.name = "Current Frame Instances"
        asso_nonquery_df.columns.name = "Nonquery Instances"

        query_frame.add_traj_score("asso_nonquery", asso_nonquery_df)

        # need frame height and width to scale boxes during post-processing
        _, h, w = query_frame.img_shape.flatten() 
        pred_boxes = model_utils.get_boxes(all_instances)
        query_boxes = pred_boxes[query_inds]  # n_k x 4
        nonquery_boxes = pred_boxes[nonquery_inds]  # n_nonquery x 4

        unique_ids = torch.unique(instance_ids)  # (n_nonquery,)

        logger.debug(f"Instance IDs: {instance_ids}")
        logger.debug(f"unique ids: {unique_ids}")

        id_inds = (
            unique_ids[None, :] == instance_ids[:, None]
        ).float()  # (n_nonquery, n_traj)

        ################################################################################

        # reweighting hyper-parameters for association -> they use 0.9

        traj_score = post_processing.weight_decay_time(
            asso_nonquery, self.decay_time, reid_features, window_size, query_ind
        )

        if self.decay_time is not None and self.decay_time > 0:
            decay_time_traj_score = pd.DataFrame(
                traj_score.clone().numpy(), columns=nonquery_inds
            )

            decay_time_traj_score.index.name = "Query Instances"
            decay_time_traj_score.columns.name = "Nonquery Instances"

            query_frame.add_traj_score("decay_time", decay_time_traj_score)
        ################################################################################
        # reduce association matrix - aggregating reference instance association scores by tracks
        # (n_query x n_nonquery) x (n_nonquery x n_traj) --> n_query x n_traj
        traj_score = torch.mm(traj_score, id_inds.cpu())  # (n_query, n_traj)

        traj_score_df = pd.DataFrame(
            traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
        )

        traj_score_df.index.name = "Current Frame Instances"
        traj_score_df.columns.name = "Unique IDs"

        query_frame.add_traj_score("traj_score", traj_score_df)
        ################################################################################
        # IOU-based post-processing; add a weighted IOU across successive frames to association scores

        # with iou -> combining with location in tracker, they set to True
        # todo -> should also work without pos_embed

        if id_inds.numel() > 0:
            # this throws error, think we need to slice?
            # last_inds = (id_inds * torch.arange(
            #    n_nonquery, device=id_inds.device)[:, None]).max(dim=0)[1] # n_traj

            last_inds = (
                id_inds * torch.arange(n_nonquery, device=id_inds.device)[:, None]
            ).max(dim=0)[
                1
            ]  # M

            last_boxes = nonquery_boxes[last_inds]  # n_traj x 4
            last_ious = post_processing._pairwise_iou(
                Boxes(query_boxes), Boxes(last_boxes)
            )  # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)

        traj_score = post_processing.weight_iou(traj_score, self.iou, last_ious.cpu())

        if self.iou is not None and self.iou != "":
            iou_traj_score = pd.DataFrame(
                traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
            )

            iou_traj_score.index.name = "Current Frame Instances"
            iou_traj_score.columns.name = "Unique IDs"

            query_frame.add_traj_score("weight_iou", iou_traj_score)
        ################################################################################
        # filters association matrix such that instances too far from each other get scores=0

        # threshold for continuing a tracking or starting a new track -> they use 1.0
        # todo -> should also work without pos_embed
        traj_score = post_processing.filter_max_center_dist(
            traj_score, self.max_center_dist, query_boxes, nonquery_boxes, id_inds, h, w
        )

        if self.max_center_dist is not None and self.max_center_dist > 0:
            max_center_dist_traj_score = pd.DataFrame(
                traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
            )

            max_center_dist_traj_score.index.name = "Current Frame Instances"
            max_center_dist_traj_score.columns.name = "Unique IDs"

            query_frame.add_traj_score("max_center_dist", max_center_dist_traj_score)

        ################################################################################
        # softmax along tracks for each instance, for interpretability
        scaled_traj_score = torch.softmax(traj_score, dim=1)
        scaled_traj_score_df = pd.DataFrame(
            scaled_traj_score.numpy(), columns=unique_ids.cpu().numpy()
        )
        scaled_traj_score_df.index.name = "Current Frame Instances"
        scaled_traj_score_df.columns.name = "Unique IDs"

        query_frame.add_traj_score("scaled", scaled_traj_score_df)
        ################################################################################

        # hungarian matching
        match_i, match_j = linear_sum_assignment((-traj_score))

        track_ids = instance_ids.new_full((n_query,), -1)
        for i, j in zip(match_i, match_j):
            # The overlap threshold is multiplied by the number of times the unique track j is matched to an
            # instance out of all instances in the window excluding the current frame.
            #
            # So if this is correct, the threshold is higher for matching an instance from the current frame
            # to an existing track if that track has already been matched several times.
            # So if an existing track in the window has been matched a lot, it gets harder to match to that track.
            thresh = (
                overlap_thresh * id_inds[:, j].sum() if mult_thresh else overlap_thresh
            )
            # if the association score for a query instance is lower than the threshold, create a new track for it
            if n_traj >= self.max_tracks or traj_score[i, j] > thresh:
                logger.debug(
                    f"Assigning instance {i} to track {j} with id {unique_ids[j]}"
                )
                track_ids[i] = unique_ids[j]
                query_frame.instances[i].track_score = scaled_traj_score[i, j].item()
        logger.debug(f"track_ids: {track_ids}")
        for i in range(n_query):
            if track_ids[i] < 0:
                logger.debug(f"Creating new track {curr_track}")
                curr_track += 1
                track_ids[i] = curr_track

        query_frame.matches = (match_i, match_j)

        for instance, track_id in zip(query_frame.instances, track_ids):
            instance.pred_track_id = track_id

        final_traj_score = pd.DataFrame(
            traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
        )
        final_traj_score.index.name = "Current Frame Instances"
        final_traj_score.columns.name = "Unique IDs"

        query_frame.add_traj_score("final", final_traj_score)
        return query_frame
