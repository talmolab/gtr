"""Module containing logic for loading datasets."""

from dreem.datasets import data_utils
from dreem.io import Frame
from torch.utils.data import Dataset
from typing import Union
import numpy as np
import torch


class BaseDataset(Dataset):
    """Base Dataset for microscopy and sleap datasets to override."""

    def __init__(
        self,
        label_files: list[str],
        vid_files: list[str],
        padding: int,
        crop_size: Union[int, list[int]],
        chunk: bool,
        clip_length: int,
        mode: str,
        augmentations: dict | None = None,
        n_chunks: int | float = 1.0,
        seed: int | None = None,
        gt_list: str | None = None,
    ):
        """Initialize Dataset.

        Args:
            label_files: a list of paths to label files.
                should at least contain detections for inference, detections + tracks for training.
            vid_files: list of paths to video files.
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train` or `val`. Determines whether this dataset is used for
                training or validation. Currently doesn't affect dataset logic
            augmentations: An optional dict mapping augmentations to parameters.
                See subclasses for details.
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
            gt_list: An optional path to .txt file containing ground truth for
                cell tracking challenge datasets.
        """
        self.vid_files = vid_files
        self.label_files = label_files
        self.padding = padding
        self.crop_size = crop_size
        self.chunk = chunk
        self.clip_length = clip_length
        self.mode = mode
        self.n_chunks = n_chunks
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        if augmentations and self.mode == "train":
            self.instance_dropout = augmentations.pop(
                "InstanceDropout", {"p": 0.0, "n": 0}
            )
            self.node_dropout = data_utils.NodeDropout(
                **augmentations.pop("NodeDropout", {"p": 0.0, "n": 0})
            )
            self.augmentations = data_utils.build_augmentations(augmentations)
        else:
            self.instance_dropout = {"p": 0.0, "n": 0}
            self.node_dropout = data_utils.NodeDropout(p=0.0, n=0)
            self.augmentations = None

        # Initialize in subclasses
        self.frame_idx = None
        self.labels = None
        self.gt_list = None

    def create_chunks(self) -> None:
        """Get indexing for data.

        Creates both indexes for selecting dataset (label_idx) and frame in
        dataset (chunked_frame_idx). If chunking is false, we index directly
        using the frame ids. Setting chunking to true creates a list of lists
        containing chunk frames for indexing. This is useful for computational
        efficiency and data shuffling. To be called by subclass __init__()
        """
        if self.chunk:
            self.chunked_frame_idx, self.label_idx = [], []
            for i, frame_idx in enumerate(self.frame_idx):
                frame_idx_split = torch.split(frame_idx, self.clip_length)
                self.chunked_frame_idx.extend(frame_idx_split)
                self.label_idx.extend(len(frame_idx_split) * [i])

            if self.n_chunks > 0 and self.n_chunks <= 1.0:
                n_chunks = int(self.n_chunks * len(self.chunked_frame_idx))

            elif self.n_chunks <= len(self.chunked_frame_idx):
                n_chunks = int(self.n_chunks)

            else:
                n_chunks = len(self.chunked_frame_idx)

            if n_chunks > 0 and n_chunks < len(self.chunked_frame_idx):
                sample_idx = np.random.choice(
                    np.arange(len(self.chunked_frame_idx)), n_chunks, replace=False
                )

                self.chunked_frame_idx = [self.chunked_frame_idx[i] for i in sample_idx]

                self.label_idx = [self.label_idx[i] for i in sample_idx]

        else:
            self.chunked_frame_idx = self.frame_idx
            self.label_idx = [i for i in range(len(self.labels))]

    def __len__(self) -> int:
        """Get the size of the dataset.

        Returns:
            the size or the number of chunks in the dataset
        """
        return len(self.chunked_frame_idx)

    def no_batching_fn(self, batch: list[Frame]) -> list[Frame]:
        """Collate function used to overwrite dataloader batching function.

        Args:
            batch: the chunk of frames to be returned

        Returns:
            The batch
        """
        return batch

    def __getitem__(self, idx: int) -> list[Frame]:
        """Get an element of the dataset.

        Args:
            idx: the index of the batch. Note this is not the index of the video
                or the frame.

        Returns:
            A list of `Frame`s in the chunk containing the metadata + instance features.
        """
        label_idx, frame_idx = self.get_indices(idx)

        return self.get_instances(label_idx, frame_idx)

    def get_indices(self, idx: int):
        """Retrieve label and frame indices given batch index.

        This method should be implemented in any subclass of the BaseDataset.

        Args:
            idx: the index of the batch.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_instances(self, label_idx: list[int], frame_idx: list[int]):
        """Build chunk of frames.

        This method should be implemented in any subclass of the BaseDataset.

        Args:
            label_idx: The index of the labels.
            frame_idx: The index of the frames.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")
