"""Script to evaluate model."""

from dreem.io import Config
from dreem.models import GTRRunner
from dreem.inference import Tracker
from omegaconf import DictConfig
from pathlib import Path

import hydra
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import sleap_io as sio
import logging

logger = logging.getLogger("dreem.inference")


@hydra.main(config_path=None, config_name=None, version_base=None)
def run(cfg: DictConfig) -> dict[int, sio.Labels]:
    """Run inference based on config file.

    Args:
        cfg: A dictconfig loaded from hydra containing checkpoint path and data
    """
    eval_cfg = Config(cfg)

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
        _ = hparams.pop("Unnamed: 0", None)

        if eval_cfg.set_hparams(hparams):
            logger.info("Updated the following hparams to the following values")
            logger.info(hparams)
    else:
        hparams = {}

    checkpoint = eval_cfg.cfg.ckpt_path

    logger.info(f"Testing model saved at {checkpoint}")
    model = GTRRunner.load_from_checkpoint(checkpoint)

    model.tracker_cfg = eval_cfg.cfg.tracker
    model.tracker = Tracker(**model.tracker_cfg)

    logger.info(f"Using the following tracker:")

    print(model.tracker)
    model.metrics["test"] = eval_cfg.cfg.runner.metrics.test
    model.persistent_tracking["test"] = eval_cfg.cfg.tracker.get(
        "persistent_tracking", False
    )
    logger.info(f"Computing the following metrics:")
    logger.info(model.metrics.test)
    model.test_results["save_path"] = eval_cfg.cfg.runner.save_path
    logger.info(f"Saving results to {model.test_results['save_path']}")

    labels_files, vid_files = eval_cfg.get_data_paths(eval_cfg.cfg.dataset.test_dataset)
    trainer = eval_cfg.get_trainer()
    for label_file, vid_file in zip(labels_files, vid_files):
        dataset = eval_cfg.get_dataset(
            label_files=[label_file], vid_files=[vid_file], mode="test"
        )
        dataloader = eval_cfg.get_dataloader(dataset, mode="test")
        metrics = trainer.test(model, dataloader)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python eval.py --config-dir=./configs --config-name=inference

    # override with params config:
    # python eval.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml

    # override with params config, and specific params:
    # python eval.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml dataset.train_dataset.padding=10
    run()
