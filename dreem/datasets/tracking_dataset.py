"""Module containing Lightning module wrapper around all other datasets."""

from dreem.datasets.cell_tracking_dataset import CellTrackingDataset
from dreem.datasets.microscopy_dataset import MicroscopyDataset
from dreem.datasets.sleap_dataset import SleapDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from copy import copy
import torch


"""
Lightning wrapper for tracking datasets
"""


class TrackingDataset(LightningDataModule):
    """Lightning dataset used to load dataloaders for train, test and validation.

    Nice for wrapping around other data formats.
    """

    def __init__(
        self,
        train_ds: SleapDataset | MicroscopyDataset | CellTrackingDataset | None = None,
        train_dl: DataLoader | None = None,
        val_ds: SleapDataset | MicroscopyDataset | CellTrackingDataset | None = None,
        val_dl: DataLoader | None = None,
        test_ds: SleapDataset | MicroscopyDataset | CellTrackingDataset | None = None,
        test_dl: DataLoader | None = None,
        splits=None,
    ):
        """Initialize tracking dataset.

        Args:
            train_ds: Sleap or Microscopy training Dataset
            train_dl: Training dataloader. Only used for overriding `train_dataloader`.
            val_ds: Sleap or Microscopy Validation set
            val_dl : Validation dataloader. Only used for overriding `val_dataloader`.
            test_ds: Sleap or Microscopy test set
            test_dl : Test dataloader. Only used for overriding `test_dataloader`.
            splits: A length 2 or 3 tuple in the order train, val, test.
                If length 3 tuple then generate train/val/test splits
                If length 2 and val_ds hasnt been created then only create a train/val dataset
                    otherwise create a val/test_ds
        """
        super().__init__()
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.val_ds = val_ds
        self.val_dl = val_dl
        self.test_ds = test_ds
        self.test_dl = test_dl
        self.splits = splits

    def setup(self, stage=None):
        """Set up lightning dataset.

        UNUSED.
        """
        if stage == "fit":
            if self.splits is not None:
                self.make_train_splits(self.splits)

        else:
            pass

    def make_train_splits(self, splits: tuple[float]) -> None:
        """Make train-val-test splits.

        If val dataset has already been initialized then just split val into val/test

        Args:
            splits: A length 2 or 3 tuple in the order train, val, test.
                If length 3 tuple then generate train/val/test splits
                If length 2 and val_ds hasnt been created then only create a train/val dataset
                    otherwise create a val/test_ds
        """
        if self.train_dl is not None:
            train_ds = self.train_dl.dataset
        else:
            train_ds = self.train_ds

        if self.val_dl is not None:
            val_ds = self.val_dl.dataset
        else:
            val_ds = self.val_ds

        if val_ds is None:
            if len(splits) == 2:
                train_frac, val_frac = splits
                test_frac = 0
            else:
                train_frac, val_frac, test_frac = splits
        else:
            val_frac, test_frac = splits
            train_frac = 0

        if train_frac != 0:
            dataset_to_split = copy(train_ds)
            dataset_to_split.augmentations = None
            dataset_size = len(dataset_to_split)
            train_inds, val_inds = train_test_split(
                range(dataset_size),
                train_size=int(dataset_size * train_frac),
                test_size=int(dataset_size * (val_frac + test_frac)),
            )
            if test_frac != 0:
                test_size = len(val_inds)
                val_inds, test_inds = train_test_split(
                    range(test_size),
                    train_size=int(test_size * (val_frac / (val_frac + test_frac))),
                    test_size=int(test_size * (test_frac / (val_frac + test_frac))),
                )
            else:
                test_inds = []

            if len(test_inds) != 0:
                test_ds = Subset(dataset_to_split, test_inds)
            else:
                test_ds = None

            val_ds = Subset(dataset_to_split, val_inds)

            train_ds = Subset(train_ds, train_inds)

        else:
            dataset_to_split = val_ds
            dataset_size = len(dataset_to_split)

            val_inds, test_inds = train_test_split(
                range(dataset_size),
                train_size=val_frac,
                test_size=test_frac,
            )

            test_ds = Subset(dataset_to_split, test_inds)
            val_ds = Subset(dataset_to_split, val_inds)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        if self.train_dl is not None:
            self.train_dl = DataLoader(
                train_ds,
                batch_size=self.train_dl.batch_size,
                shuffle=True,
                num_workers=self.train_dl.num_workers,
                collate_fn=train_ds.dataset.no_batching_fn,
                pin_memory=self.train_dl.pin_memory,
                drop_last=self.train_dl.drop_last,
                timeout=self.train_dl.timeout,
                worker_init_fn=self.train_dl.worker_init_fn,
                multiprocessing_context=self.train_dl.multiprocessing_context,
                generator=self.train_dl.generator,
                prefetch_factor=self.train_dl.prefetch_factor,
                persistent_workers=self.train_dl.persistent_workers,
                pin_memory_device=self.train_dl.pin_memory_device,
            )
        if self.val_dl is not None:
            self.val_dl = DataLoader(
                val_ds,
                batch_size=self.val_dl.batch_size,
                shuffle=False,
                num_workers=self.val_dl.num_workers,
                collate_fn=val_ds.dataset.no_batching_fn,
                pin_memory=self.val_dl.pin_memory,
                drop_last=self.val_dl.drop_last,
                timeout=self.val_dl.timeout,
                worker_init_fn=self.val_dl.worker_init_fn,
                multiprocessing_context=self.val_dl.multiprocessing_context,
                generator=self.val_dl.generator,
                prefetch_factor=self.val_dl.prefetch_factor,
                persistent_workers=self.val_dl.persistent_workers,
                pin_memory_device=self.val_dl.pin_memory_device,
            )
        if self.test_dl is not None:
            self.test_dl = DataLoader(
                test_ds,
                batch_size=self.test_dl.batch_size,
                shuffle=False,
                num_workers=self.test_dl.num_workers,
                collate_fn=test_ds.dataset.no_batching_fn,
                pin_memory=self.test_dl.pin_memory,
                drop_last=self.test_dl.drop_last,
                timeout=self.test_dl.timeout,
                worker_init_fn=self.test_dl.worker_init_fn,
                multiprocessing_context=self.test_dl.multiprocessing_context,
                generator=self.test_dl.generator,
                prefetch_factor=self.test_dl.prefetch_factor,
                persistent_workers=self.test_dl.persistent_workers,
                pin_memory_device=self.test_dl.pin_memory_device,
            )

    def train_dataloader(self) -> DataLoader:
        """Get train_dataloader.

        Returns: The Training Dataloader.
        """
        if self.train_dl is None and self.train_ds is None:
            return None
        elif self.train_dl is None:

            if isinstance(self.train_ds, Subset):
                batching_fn = self.train_ds.dataset.no_batching_fn
            else:
                batching_fn = self.train_ds.no_batching_fn
            return DataLoader(
                self.train_ds,
                batch_size=1,
                shuffle=True,
                pin_memory=False,
                collate_fn=batching_fn,
                num_workers=0,
                generator=None,
            )
        else:
            return self.train_dl

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader.

        Returns: The validation dataloader.
        """
        if self.val_dl is None and self.val_ds is None:
            return None
        elif self.val_dl is None:
            if isinstance(self.val_ds, Subset):
                batching_fn = self.val_ds.dataset.no_batching_fn
            else:
                batching_fn = self.val_ds.no_batching_fn

            return DataLoader(
                self.val_ds,
                batch_size=1,
                shuffle=False,
                pin_memory=0,
                collate_fn=batching_fn,
                num_workers=False,
                generator=None,
            )
        else:
            return self.val_dl

    def test_dataloader(self) -> DataLoader:
        """Get test_dataloader.

        Returns: The test dataloader
        """
        if self.test_dl is None and self.test_ds is None:
            return None
        elif self.test_dl is None:
            if isinstance(self.test_ds, Subset):
                batching_fn = self.test_ds.dataset.no_batching_fn
            else:
                batching_fn = self.test_ds.no_batching_fn

            return DataLoader(
                self.test_ds,
                batch_size=1,
                shuffle=False,
                pin_memory=0,
                collate_fn=batching_fn,
                num_workers=False,
                generator=None,
            )
        else:
            return self.test_dl
