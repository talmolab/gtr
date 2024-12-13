# to implement - config class that handles getters/setters
"""Data structures for handling config parsing."""

from __future__ import annotations
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Iterable
from pathlib import Path
import logging
import glob
import pytorch_lightning as pl
import torch

logger = logging.getLogger("dreem.io")


class Config:
    """Class handling loading components based on config params."""

    def __init__(self, cfg: DictConfig, params_cfg: DictConfig | None = None):
        """Initialize the class with config from hydra/omega conf.

        First uses `base_param` file then overwrites with specific `params_config`.

        Args:
            cfg: The `DictConfig` containing all the hyperparameters needed for
                training/evaluation.
            params_cfg: The `DictConfig` containing subset of hyperparameters to override.
                training/evaluation
        """
        base_cfg = cfg
        logger.info(f"Base Config: {cfg}")

        if "params_config" in cfg:
            params_cfg = OmegaConf.load(cfg.params_config)

        if params_cfg:
            logger.info(f"Overwriting base config with {params_cfg}")
            with open_dict(base_cfg):
                self.cfg = OmegaConf.merge(base_cfg, params_cfg)  # merge configs
        else:
            self.cfg = cfg

        OmegaConf.set_struct(self.cfg, False)

        self._vid_files = {}

    def __repr__(self):
        """Object representation of config class."""
        return f"Config({self.cfg})"

    def __str__(self):
        """Return a string representation of config class."""
        return f"Config({self.cfg})"

    @classmethod
    def from_yaml(cls, base_cfg_path: str, params_cfg_path: str | None = None) -> None:
        """Load config directly from yaml.

        Args:
            base_cfg_path: path to base config file.
            params_cfg_path: path to override params.
        """
        base_cfg = OmegaConf.load(base_cfg_path)
        params_cfg = OmegaConf.load(params_cfg_path) if params_cfg_path else None
        return cls(base_cfg, params_cfg)

    def set_hparams(self, hparams: dict) -> bool:
        """Setter function for overwriting specific hparams.

        Useful for changing 1 or 2 hyperparameters such as dataset.

        Args:
            hparams: A dict containing the hyperparameter to be overwritten and
                the value to be changed

        Returns:
            `True` if config is successfully updated, `False` otherwise
        """
        if hparams == {} or hparams is None:
            logger.warning("Nothing to update!")
            return False
        for hparam, val in hparams.items():
            try:
                OmegaConf.update(self.cfg, hparam, val)
            except Exception as e:
                logger.exception(f"Failed to update {hparam} to {val} due to {e}")
                return False
        return True

    def get(self, key: str, default=None, cfg: dict = None):
        """Get config item.

        Args:
            key: key of item to return
            default: default value to return if key is missing.
            cfg: the config dict from which to retrieve an item
        """
        if cfg is None:
            cfg = self.cfg

        param = cfg.get(key, default)

        if isinstance(param, DictConfig):
            param = OmegaConf.to_container(param, resolve=True)

        return param

    def get_model(self) -> "GlobalTrackingTransformer":
        """Getter for gtr model.

        Returns:
            A global tracking transformer with parameters indicated by cfg
        """
        from dreem.models import GlobalTrackingTransformer, GTRRunner

        model_params = self.get("model", {})

        ckpt_path = model_params.pop("ckpt_path", None)

        if ckpt_path is not None and len(ckpt_path) > 0:
            return GTRRunner.load_from_checkpoint(ckpt_path).model

        return GlobalTrackingTransformer(**model_params)

    def get_tracker_cfg(self) -> dict:
        """Getter for tracker config params.

        Returns:
            A dict containing the init params for `Tracker`.
        """
        return self.get("tracker", {})

    def get_gtr_runner(self, ckpt_path: str | None = None) -> "GTRRunner":
        """Get lightning module for training, validation, and inference.

        Args:
            ckpt_path: path to checkpoint for override

        Returns:
            a gtr runner model
        """
        from dreem.models import GTRRunner

        keys = ["tracker", "optimizer", "scheduler", "loss", "runner", "model"]
        args = [key + "_cfg" if key != "runner" else key for key in keys]

        params = {}
        for key, arg in zip(keys, args):
            sub_params = self.get(key, {})

            if len(sub_params) == 0:
                logger.warning(
                    f"`{key}` not found in config or is empty. Using defaults for {arg}!"
                )

            if key == "runner":
                runner_params = sub_params
                for k, v in runner_params.items():
                    params[k] = v
            else:
                params[arg] = sub_params

        ckpt_path = params["model_cfg"].pop("ckpt_path", None)

        if ckpt_path is not None and ckpt_path != "":
            model = GTRRunner.load_from_checkpoint(
                ckpt_path, tracker_cfg=params["tracker_cfg"], **runner_params
            )

        else:
            model = GTRRunner(**params)

        return model

    def get_data_paths(self, data_cfg: dict) -> tuple[list[str], list[str]]:
        """Get file paths from directory.

        Args:
            data_cfg: Config for the dataset containing "dir" key.

        Returns:
            lists of labels file paths and video file paths respectively
        """
        dir_cfg = data_cfg.pop("dir", None)
        label_files = vid_files = None

        if dir_cfg:
            labels_suff = dir_cfg.get("labels_suffix")
            vid_suff = dir_cfg.get("vid_suffix")
            if labels_suff is None or vid_suff is None:
                raise KeyError(
                    f"Must provide a labels suffix and vid suffix to search for but found {labels_suff} and {vid_suff}!"
                )
            list_dir_path = dir_cfg.get("path", ".")
            if not isinstance(list_dir_path, list):
                list_dir_path = [list_dir_path]
            label_files = []
            vid_files = []
            for dir_path in list_dir_path:
                logger.debug(f"Searching `{dir_path}` directory")

                labels_path = f"{dir_path}/*{labels_suff}"
                vid_path = f"{dir_path}/*{vid_suff}"
                logger.debug(f"Searching for labels matching {labels_path}")
                label_files.extend(glob.glob(labels_path))
                logger.debug(f"Searching for videos matching {vid_path}")
                vid_files.extend(glob.glob(vid_path))

            logger.debug(f"Found {len(label_files)} labels and {len(vid_files)} videos")

        else:
            if "slp_files" in data_cfg:
                label_files = data_cfg["slp_files"]
                vid_files = data_cfg["video_files"]
            elif "tracks" in data_cfg or "source" in data_cfg:
                label_files = data_cfg["tracks"]
                vid_files = data_cfg["videos"]
            elif "raw_images" in data_cfg:
                label_files = data_cfg["gt_images"]
                vid_files = data_cfg["raw_images"]

        return label_files, vid_files

    def get_dataset(
        self,
        mode: str,
        label_files: list[str] | None = None,
        vid_files: list[str | list[str]] = None,
    ) -> "SleapDataset" | "MicroscopyDataset" | "CellTrackingDataset" | None:
        """Getter for datasets.

        Args:
            mode: [None, "train", "test", "val"]. Indicates whether to use
                train, val, or test params for dataset
            label_files: path to label_files for override
            vid_files: path to vid_files for override

        Returns:
            Either a `SleapDataset` or `MicroscopyDataset` with params indicated by cfg
        """
        from dreem.datasets import MicroscopyDataset, SleapDataset, CellTrackingDataset

        dataset_params = self.get("dataset")
        if dataset_params is None:
            raise KeyError("`dataset` key is missing from cfg!")

        if mode.lower() == "train":
            dataset_params = self.get("train_dataset", {}, dataset_params)
        elif mode.lower() == "val":
            dataset_params = self.get("val_dataset", {}, dataset_params)
        elif mode.lower() == "test":
            dataset_params = self.get("test_dataset", {}, dataset_params)
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )

        if "dir" in dataset_params:
            self.data_dirs = dataset_params["dir"]["path"]
        else:
            self.data_dirs = []

        if label_files is None or vid_files is None:
            label_files, vid_files = self.get_data_paths(dataset_params)
        # todo: handle this better
        if "slp_files" in dataset_params:
            if label_files is not None:
                dataset_params["slp_files"] = label_files
            if vid_files is not None:
                dataset_params["video_files"] = vid_files

            dataset_params["data_dirs"] = self.data_dirs
            # TODO: handle this better!
            if "model" in self.cfg:
                if (
                    "encoder_type" in self.cfg["model"]["encoder_cfg"]
                    and self.cfg["model"]["encoder_cfg"]["encoder_type"] == "descriptor"
                ):
                    dataset_params["normalize_image"] = False

            self.data_paths = (mode, vid_files)

            return SleapDataset(**dataset_params)

        elif "tracks" in dataset_params or "source" in dataset_params:
            if label_files is not None:
                dataset_params["tracks"] = label_files
            if vid_files is not None:
                dataset_params["videos"] = vid_files
            return MicroscopyDataset(**dataset_params)

        elif "raw_images" in dataset_params:
            if label_files is not None:
                dataset_params["gt_images"] = label_files
            if vid_files is not None:
                dataset_params["raw_images"] = vid_files
            return CellTrackingDataset(**dataset_params)

        else:
            raise ValueError(
                "Could not resolve dataset type from Config! Please include \
                either `slp_files` or `tracks`/`source`"
            )
        if len(dataset) == 0:
            logger.warn(f"Length of {mode} dataset is {len(dataset)}! Returning None")
            return None
        return dataset

    @property
    def data_paths(self):
        """Get data paths."""
        return self._vid_files

    @data_paths.setter
    def data_paths(self, paths: tuple[str, list[str]]):
        """Set data paths.

        Args:
            paths: A tuple containing (mode, vid_files)
        """
        mode, vid_files = paths
        self._vid_files[mode] = vid_files

    def get_dataloader(
        self,
        dataset: "SleapDataset" | "MicroscopyDataset" | "CellTrackingDataset" | None,
        mode: str,
    ) -> torch.utils.data.DataLoader | None:
        """Getter for dataloader.

        Args:
            dataset: the Sleap or Microscopy Dataset used to initialize the dataloader
            mode: either ["train", "val", or "test"] indicates which dataset
                config to use

        Returns:
            A torch dataloader for `dataset` with parameters configured as specified
        """
        dataloader_params = self.get("dataloader", {})
        if mode.lower() == "train":
            dataloader_params = self.get("train_dataloader", {}, dataloader_params)
        elif mode.lower() == "val":
            dataloader_params = self.get("val_dataloader", {}, dataloader_params)
        elif mode.lower() == "test":
            dataloader_params = self.get("test_dataloader", {}, dataloader_params)
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )
        if dataloader_params.get("num_workers", 0) > 0:
            # prevent too many open files error
            pin_memory = True
            torch.multiprocessing.set_sharing_strategy("file_system")
        else:
            pin_memory = False

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            pin_memory=pin_memory,
            collate_fn=dataset.no_batching_fn,
            **dataloader_params,
        )

        if len(dataloader) == 0:
            logger.warn(
                f"Length of {mode} dataloader is {len(dataloader)}! Returning `None`"
            )
            return None
        return dataloader

    def get_optimizer(self, params: Iterable) -> torch.optim.Optimizer:
        """Getter for optimizer.

        Args:
            params: iterable of model parameters to optimize or dicts defining
                parameter groups

        Returns:
            A torch Optimizer with specified params
        """
        from dreem.models.model_utils import init_optimizer

        optimizer_params = self.get("optimizer")

        return init_optimizer(params, optimizer_params)

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Getter for lr scheduler.

        Args:
            optimizer: The optimizer to wrap the scheduler around

        Returns:
            A torch learning rate scheduler with specified params
        """
        from dreem.models.model_utils import init_scheduler

        lr_scheduler_params = self.get("scheduler")

        if lr_scheduler_params is None:
            logger.warning(
                "`scheduler` key not found in cfg or is empty. No scheduler will be returned!"
            )
            return None
        return init_scheduler(optimizer, lr_scheduler_params)

    def get_loss(self) -> "dreem.training.losses.AssoLoss":
        """Getter for loss functions.

        Returns:
            An AssoLoss with specified params
        """
        from dreem.training.losses import AssoLoss

        loss_params = self.get("loss", {})

        if len(loss_params) == 0:
            logger.warning(
                "`loss` key not found in cfg. Using default params for `AssoLoss`"
            )

        return AssoLoss(**loss_params)

    def get_logger(self) -> pl.loggers.Logger:
        """Getter for logging callback.

        Returns:
            A Logger with specified params
        """
        from dreem.models.model_utils import init_logger

        logger_params = self.get("logging", {})
        if len(logger_params) == 0:
            logger.warning(
                "`logging` key not found in cfg. No logger will be configured!"
            )

        return init_logger(
            logger_params, OmegaConf.to_container(self.cfg, resolve=True)
        )

    def get_early_stopping(self) -> pl.callbacks.EarlyStopping:
        """Getter for lightning early stopping callback.

        Returns:
            A lightning early stopping callback with specified params
        """
        early_stopping_params = self.get("early_stopping", None)

        if early_stopping_params is None:
            logger.warning(
                "`early_stopping` was not found in cfg or was `null`. Early stopping will not be used!"
            )
            return None
        elif len(early_stopping_params) == 0:
            logger.warning("`early_stopping` cfg is empty! Using defaults")
        return pl.callbacks.EarlyStopping(**early_stopping_params)

    def get_checkpointing(self) -> pl.callbacks.ModelCheckpoint:
        """Getter for lightning checkpointing callback.

        Returns:
            A lightning checkpointing callback with specified params
        """
        # convert to dict to enable extracting/removing params
        checkpoint_params = self.get("checkpointing", {})
        logging_params = self.get("logging", {})

        dirpath = checkpoint_params.pop("dirpath", None)

        if dirpath is None:
            dirpath = f"./models/{self.get('group', '', logging_params)}/{self.get('name', '', logging_params)}"

        dirpath = Path(dirpath).resolve()
        if not Path(dirpath).exists():
            try:
                Path(dirpath).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.exception(
                    f"Cannot create a new folder!. Check the permissions to {dirpath}. \n {e}"
                )

        _ = checkpoint_params.pop("dirpath", None)
        monitor = checkpoint_params.pop("monitor", ["val_loss"])
        checkpointers = []

        logger.info(
            f"Saving checkpoints to `{dirpath}` based on the following metrics: {monitor}"
        )
        if len(checkpoint_params) == 0:
            logger.warning(
                """`checkpointing` key was not found in cfg or was empty!
                Configuring checkpointing to use default params!"""
            )

        for metric in monitor:
            checkpointer = pl.callbacks.ModelCheckpoint(
                monitor=metric,
                dirpath=dirpath,
                filename=f"{{epoch}}-{{{metric}}}",
                **checkpoint_params,
            )
            checkpointer.CHECKPOINT_NAME_LAST = f"{{epoch}}-final-{{{metric}}}"
            checkpointers.append(checkpointer)
        return checkpointers

    def get_trainer(
        self,
        callbacks: list[pl.callbacks.Callback] | None = None,
        logger: pl.loggers.WandbLogger | None = None,
        devices: int = 1,
        accelerator: str = "auto",
    ) -> pl.Trainer:
        """Getter for the lightning trainer.

        Args:
            callbacks: a list of lightning callbacks preconfigured to be used
                for training
            logger: the Wandb logger used for logging during training
            devices: The number of gpus to be used. 0 means cpu
            accelerator: either "gpu" or "cpu" specifies which device to use

        Returns:
            A lightning Trainer with specified params
        """
        trainer_params = self.get("trainer", {})
        profiler = trainer_params.pop("profiler", None)
        if len(trainer_params) == 0:
            print(
                "`trainer` key was not found in cfg or was empty. Using defaults for `pl.Trainer`!"
            )

        if "accelerator" not in trainer_params:
            trainer_params["accelerator"] = accelerator
        if "devices" not in trainer_params:
            trainer_params["devices"] = devices

        map_profiler = {
            "advanced": pl.profilers.AdvancedProfiler,
            "simple": pl.profilers.SimpleProfiler,
            "pytorch": pl.profilers.PyTorchProfiler,
            "passthrough": pl.profilers.PassThroughProfiler,
            "xla": pl.profilers.XLAProfiler,
        }

        if profiler:
            if profiler in map_profiler:
                profiler = map_profiler[profiler](filename="profile")
            else:
                raise ValueError(
                    f"Profiler {profiler} not supported! Please use one of {list(map_profiler.keys())}"
                )

        return pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            profiler=profiler,
            **trainer_params,
        )
