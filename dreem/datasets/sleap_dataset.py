"""Module containing logic for loading sleap datasets."""

import albumentations as A
import torch
import imageio
import numpy as np
import sleap_io as sio
import random
from pathlib import Path
import logging
from typing import Union, Optional
from dreem.io import Instance, Frame
from dreem.datasets import data_utils, BaseDataset
from torchvision.transforms import functional as tvf

logger = logging.getLogger("dreem.datasets")


class SleapDataset(BaseDataset):
    """Dataset for loading animal behavior data from sleap."""

    def __init__(
        self,
        slp_files: list[str],
        video_files: list[str],
        data_dirs: Optional[list[str]] = None,
        padding: int = 5,
        crop_size: Union[int, list[int]] = 128,
        anchors: int | list[str] | str = "",
        chunk: bool = True,
        clip_length: int = 500,
        mode: str = "train",
        handle_missing: str = "centroid",
        augmentations: dict | None = None,
        n_chunks: int | float = 1.0,
        seed: int | None = None,
        verbose: bool = False,
        normalize_image: bool = True,
    ):
        """Initialize SleapDataset.

        Args:
            slp_files: a list of .slp files storing tracking annotations
            video_files: a list of paths to video files
            data_dirs: a path, or a list of paths to data directories. If provided, crop_size should be a list of integers
                with the same length as data_dirs.
            padding: amount of padding around object crops
            crop_size: the size of the object crops. Can be either:
                - An integer specifying a single crop size for all objects
                - A list of integers specifying different crop sizes for different data directories
            anchors: One of:
                        * a string indicating a single node to center crops around
                        * a list of skeleton node names to be used as the center of crops
                        * an int indicating the number of anchors to randomly select
                    If unavailable then crop around the midpoint between all visible anchors.
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train`, `val`, or `test`. Determines whether this dataset is used for
                training, validation/testing/inference.
            handle_missing: how to handle missing single nodes. one of `["drop", "ignore", "centroid"]`.
                            if "drop" then we dont include instances which are missing the `anchor`.
                            if "ignore" then we use a mask instead of a crop and nan centroids/bboxes.
                            if "centroid" then we default to the pose centroid as the node to crop around.
            augmentations: An optional dict mapping augmentations to parameters. The keys
                should map directly to augmentation classes in albumentations. Example:
                    augmentations = {
                        'Rotate': {'limit': [-90, 90], 'p': 0.5},
                        'GaussianBlur': {'blur_limit': (3, 7), 'sigma_limit': 0, 'p': 0.2},
                        'RandomContrast': {'limit': 0.2, 'p': 0.6}
                    }
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
            verbose: boolean representing whether to print
            normalize_image: whether to normalize the image to [0, 1]
        """
        super().__init__(
            slp_files,
            video_files,
            padding,
            crop_size,
            chunk,
            clip_length,
            mode,
            augmentations,
            n_chunks,
            seed,
        )

        self.slp_files = slp_files
        self.data_dirs = (
            data_dirs  # empty list, list of paths, or string of single path
        )
        self.video_files = video_files
        self.padding = padding
        self.crop_size = crop_size
        self.chunk = chunk
        self.clip_length = clip_length
        self.mode = mode.lower()
        self.handle_missing = handle_missing.lower()
        self.n_chunks = n_chunks
        self.seed = seed
        self.normalize_image = normalize_image
        if self.data_dirs is None:
            self.data_dirs = []
        if isinstance(anchors, int):
            self.anchors = anchors
        elif isinstance(anchors, str):
            self.anchors = [anchors]
        else:
            self.anchors = anchors

        if not isinstance(self.data_dirs, list):
            self.data_dirs = [self.data_dirs]

        if not isinstance(self.crop_size, list):
            # make a list so its handled consistently if multiple crops are used
            if len(self.data_dirs) > 0:  # for test mode, data_dirs is []
                self.crop_size = [self.crop_size] * len(self.data_dirs)
            else:
                self.crop_size = [self.crop_size]

        if len(self.data_dirs) > 0 and len(self.crop_size) != len(self.data_dirs):
            raise ValueError(
                f"If a list of crop sizes or data directories are given,"
                f"they must have the same length but got {len(self.crop_size)} "
                f"and {len(self.data_dirs)}"
            )

        if (
            isinstance(self.anchors, list) and len(self.anchors) == 0
        ) or self.anchors == 0:
            raise ValueError(f"Must provide at least one anchor but got {self.anchors}")

        self.verbose = verbose

        # if self.seed is not None:
        #     np.random.seed(self.seed)
        self.labels = [sio.load_slp(slp_file) for slp_file in self.slp_files]
        self.videos = [imageio.get_reader(vid_file) for vid_file in self.vid_files]
        # do we need this? would need to update with sleap-io

        # for label in self.labels:
        # label.remove_empty_instances(keep_empty_frames=False)

        self.frame_idx = [torch.arange(len(labels)) for labels in self.labels]
        # Method in BaseDataset. Creates label_idx and chunked_frame_idx to be
        # used in call to get_instances()
        self.create_chunks()

    def get_indices(self, idx: int) -> tuple:
        """Retrieve label and frame indices given batch index.

        Args:
            idx: the index of the batch.
        """
        return self.label_idx[idx], self.chunked_frame_idx[idx]

    def get_instances(self, label_idx: list[int], frame_idx: list[int]) -> list[Frame]:
        """Get an element of the dataset.

        Args:
            label_idx: index of the labels
            frame_idx: index of the frames

        Returns:
            A list of `dreem.io.Frame` objects containing metadata and instance data for the batch/clip.

        """
        video = self.labels[label_idx]

        video_name = self.video_files[label_idx]

        # get the correct crop size based on the video
        video_par_path = Path(video_name).parent
        if len(self.data_dirs) > 0:
            crop_size = self.crop_size[0]
            for j, data_dir in enumerate(self.data_dirs):
                if Path(data_dir) == video_par_path:
                    crop_size = self.crop_size[j]
                    break
        else:
            crop_size = self.crop_size[0]

        vid_reader = self.videos[label_idx]

        # img = vid_reader.get_data(0)

        skeleton = video.skeletons[-1]

        frames = []
        for i, frame_ind in enumerate(frame_idx):
            (
                instances,
                gt_track_ids,
                poses,
                shown_poses,
                point_scores,
                instance_score,
            ) = ([], [], [], [], [], [])

            frame_ind = int(frame_ind)

            lf = video[frame_ind]

            try:
                img = vid_reader.get_data(int(lf.frame_idx))
            except IndexError as e:
                logger.warning(
                    f"Could not read frame {frame_ind} from {video_name} due to {e}"
                )
                continue

            if len(img.shape) == 2:
                img = img.expand_dims(-1)
            h, w, c = img.shape

            if c == 1:
                img = np.concatenate(
                    [img, img, img], axis=-1
                )  # convert to grayscale to rgb

            if np.issubdtype(img.dtype, np.integer):  # convert int to float
                img = img.astype(np.float32)
                if self.normalize_image:
                    img = img / 255

            n_instances_dropped = 0

            gt_instances = lf.instances
            if self.mode == "train":
                np.random.shuffle(gt_instances)

            for instance in gt_instances:
                if (
                    np.random.uniform() < self.instance_dropout["p"]
                    and n_instances_dropped < self.instance_dropout["n"]
                ):
                    n_instances_dropped += 1
                    continue

                if instance.track is not None:
                    gt_track_id = video.tracks.index(instance.track)
                else:
                    gt_track_id = -1
                gt_track_ids.append(gt_track_id)

                poses.append(
                    dict(
                        zip(
                            [n.name for n in instance.skeleton.nodes],
                            [[p.x, p.y] for p in instance.points.values()],
                        )
                    )
                )

                shown_poses = [
                    {
                        key: val
                        for key, val in instance.items()
                        if not np.isnan(val).any()
                    }
                    for instance in poses
                ]

                point_scores.append(
                    np.array(
                        [
                            (
                                point.score
                                if isinstance(point, sio.PredictedPoint)
                                else 1.0
                            )
                            for point in instance.points.values()
                        ]
                    )
                )
                if isinstance(instance, sio.PredictedInstance):
                    instance_score.append(instance.score)
                else:
                    instance_score.append(1.0)
            # augmentations
            if self.augmentations is not None:
                for transform in self.augmentations:
                    if isinstance(transform, A.CoarseDropout):
                        transform.fill_value = random.randint(0, 255)

                if shown_poses:
                    keypoints = np.vstack([list(s.values()) for s in shown_poses])

                else:
                    keypoints = []

                augmented = self.augmentations(image=img, keypoints=keypoints)

                img, aug_poses = augmented["image"], augmented["keypoints"]

                aug_poses = [
                    arr
                    for arr in np.split(
                        np.array(aug_poses),
                        np.array([len(s) for s in shown_poses]).cumsum(),
                    )
                    if arr.size != 0
                ]

                aug_poses = [
                    dict(zip(list(pose_dict.keys()), aug_pose_arr.tolist()))
                    for aug_pose_arr, pose_dict in zip(aug_poses, shown_poses)
                ]

                _ = [
                    pose.update(aug_pose)
                    for pose, aug_pose in zip(shown_poses, aug_poses)
                ]

            img = tvf.to_tensor(img)

            for j in range(len(gt_track_ids)):
                pose = shown_poses[j]

                """Check for anchor"""
                crops = []
                boxes = []
                centroids = {}

                if isinstance(self.anchors, int):
                    anchors_to_choose = list(pose.keys()) + ["midpoint"]
                    anchors = np.random.choice(anchors_to_choose, self.anchors)
                else:
                    anchors = self.anchors

                dropped_anchors = self.node_dropout(anchors)

                for anchor in anchors:
                    if anchor in dropped_anchors:
                        centroid = np.array([np.nan, np.nan])

                    elif anchor == "midpoint" or anchor == "centroid":
                        centroid = np.nanmean(np.array(list(pose.values())), axis=0)

                    elif anchor in pose:
                        centroid = np.array(pose[anchor])
                        if np.isnan(centroid).any():
                            centroid = np.array([np.nan, np.nan])

                    elif (
                        anchor not in pose
                        and len(anchors) == 1
                        and self.handle_missing == "centroid"
                    ):
                        anchor = "midpoint"
                        centroid = np.nanmean(np.array(list(pose.values())), axis=0)

                    else:
                        centroid = np.array([np.nan, np.nan])

                    if np.isnan(centroid).all():
                        bbox = torch.tensor([np.nan, np.nan, np.nan, np.nan])

                    else:
                        bbox = data_utils.pad_bbox(
                            data_utils.get_bbox(centroid, crop_size),
                            padding=self.padding,
                        )

                    if bbox.isnan().all():
                        crop = torch.zeros(
                            c,
                            crop_size + 2 * self.padding,
                            crop_size + 2 * self.padding,
                            dtype=img.dtype,
                        )
                    else:
                        crop = data_utils.crop_bbox(img, bbox)

                    crops.append(crop)
                    centroids[anchor] = centroid
                    boxes.append(bbox)

                if len(crops) > 0:
                    crops = torch.concat(crops, dim=0)

                if len(boxes) > 0:
                    boxes = torch.stack(boxes, dim=0)

                if self.handle_missing == "drop" and boxes.isnan().any():
                    continue

                instance = Instance(
                    gt_track_id=gt_track_ids[j],
                    pred_track_id=-1,
                    crop=crops,
                    centroid=centroids,
                    bbox=boxes,
                    skeleton=skeleton,
                    pose=poses[j],
                    point_scores=point_scores[j],
                    instance_score=instance_score[j],
                )

                instances.append(instance)

            frame = Frame(
                video_id=label_idx,
                frame_id=frame_ind,
                vid_file=video_name,
                img_shape=img.shape,
                instances=instances,
            )
            frames.append(frame)

        return frames

    def __del__(self):
        """Handle file closing before garbage collection."""
        for reader in self.videos:
            reader.close()