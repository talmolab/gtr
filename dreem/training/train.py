"""Training script for training model.

Used for training a single model or deploying a batch train job on RUNAI CLI
"""

from dreem.io import Config
from dreem.datasets import TrackingDataset
from dreem.datasets.data_utils import view_training_batch
from multiprocessing import cpu_count
from omegaconf import DictConfig

import os
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import logging

logger = logging.getLogger("training")


@hydra.main(config_path=None, config_name=None, version_base=None)
def run(cfg: DictConfig):
    """Train model based on config.

    Handles all config parsing and initialization then calls `trainer.train()`.
    If `batch_config` is included then run will be assumed to be a batch job.

    Args:
        cfg: The config dict parsed by `hydra`
    """
    torch.set_float32_matmul_precision("medium")
    train_cfg = Config(cfg)

    # update with parameters for batch train job
    if "batch_config" in cfg.keys():
        try:
            index = int(os.environ["POD_INDEX"])
        except KeyError as e:
            index = int(
                input(f"{e}. Assuming single run!\nPlease input task index to run:")
            )

        hparams_df = pd.read_csv(cfg.batch_config)
        hparams = hparams_df.iloc[index].to_dict()

        if train_cfg.set_hparams(hparams):
            logger.debug("Updated the following hparams to the following values")
            logger.debug(hparams)
    else:
        hparams = {}
    logging.getLogger().setLevel(level=cfg.get("log_level", "INFO").upper())
    logger.info(f"Final train config: {train_cfg}")

    model = train_cfg.get_model()
    train_dataset = train_cfg.get_dataset(mode="train")
    train_dataloader = train_cfg.get_dataloader(train_dataset, mode="train")

    val_dataset = train_cfg.get_dataset(mode="val")
    val_dataloader = train_cfg.get_dataloader(val_dataset, mode="val")

    test_dataset = train_cfg.get_dataset(mode="test")
    test_dataloader = train_cfg.get_dataloader(test_dataset, mode="test")

    dataset = TrackingDataset(
        train_dl=train_dataloader, val_dl=val_dataloader, test_dl=test_dataloader
    )

    if cfg.view_batch.enable:
        instances = next(iter(train_dataset))
        view_training_batch(instances, num_frames=cfg.view_batch.num_frames)

        if cfg.view_batch.no_train:
            return

    model = train_cfg.get_gtr_runner()  # TODO see if we can use torch.compile()

    run_logger = train_cfg.get_logger()

    if run_logger is not None and isinstance(run_logger, pl.loggers.wandb.WandbLogger):
        data_paths = train_cfg.data_paths
        flattened_paths = [
            [item] for sublist in data_paths.values() for item in sublist
        ]
        run_logger.log_text(
            "training_files", columns=["data_paths"], data=flattened_paths
        )

    callbacks = []
    _ = callbacks.extend(train_cfg.get_checkpointing())
    _ = callbacks.append(pl.callbacks.LearningRateMonitor())

    early_stopping = train_cfg.get_early_stopping()
    if early_stopping is not None:
        callbacks.append(early_stopping)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if torch.cuda.is_available() else cpu_count()

    trainer = train_cfg.get_trainer(
        callbacks,
        run_logger,
        accelerator=accelerator,
        devices=devices,
    )

    trainer.fit(model, dataset)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py --config-dir=./configs --config-name=base

    # override with params config:
    # python train.py --config-dir=./configs --config-name=base +params_config=configs/params.yaml

    # override with params config, and specific params:
    # python train.py --config-dir=./configs --config-name=base +params_config=configs/params.yaml model.norm=True model.decoder_self_attn=True dataset.padding=10

    # deploy batch train job:
    # python train.py --config-dir=./configs --config-name=base +batch_config=test_batch_train.csv

    run()
