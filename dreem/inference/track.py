"""Script to run inference and get out tracks."""

from dreem.io import Config
from dreem.inference import Tracker
from dreem.models import GTRRunner
from omegaconf import DictConfig
from pathlib import Path
from datetime import datetime

import hydra
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import sleap_io as sio
import logging


logger = logging.getLogger("dreem.inference")


def get_timestamp() -> str:
    """Get current timestamp.

    Returns:
        the current timestamp in /m/d/y-H:M:S format
    """
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    return date_time


def export_trajectories(
    frames_pred: list["dreem.io.Frame"], save_path: str | None = None
) -> pd.DataFrame:
    """Convert trajectories to data frame and save as .csv.

    Args:
        frames_pred: A list of Frames with predicted track ids.
        save_path: The path to save the predicted trajectories to.

    Returns:
        A dictionary containing the predicted track id and centroid coordinates for each instance in the video.
    """
    save_dict = {}
    frame_ids = []
    X, Y = [], []
    pred_track_ids = []
    track_scores = []
    for frame in frames_pred:
        for i, instance in enumerate(frame.instances):
            frame_ids.append(frame.frame_id.item())
            bbox = instance.bbox.squeeze()
            y = (bbox[2] + bbox[0]) / 2
            x = (bbox[3] + bbox[1]) / 2
            X.append(x.item())
            Y.append(y.item())
            track_scores.append(instance.track_score)
            pred_track_ids.append(instance.pred_track_id.item())

    save_dict["Frame"] = frame_ids
    save_dict["X"] = X
    save_dict["Y"] = Y
    save_dict["Pred_track_id"] = pred_track_ids
    save_dict["Track_score"] = track_scores
    save_df = pd.DataFrame(save_dict)
    if save_path:
        save_df.to_csv(save_path, index=False)
    return save_df


def track(
    model: GTRRunner, trainer: pl.Trainer, dataloader: torch.utils.data.DataLoader
) -> list[pd.DataFrame]:
    """Run Inference.

    Args:
        model: GTRRunner model loaded from checkpoint used for inference
        trainer: lighting Trainer object used for handling inference log.
        dataloader: dataloader containing inference data

    Return:
        List of DataFrames containing prediction results for each video
    """
    preds = trainer.predict(model, dataloader)
    pred_slp = []
    tracks = {}
    for batch in preds:
        for frame in batch:
            if frame.frame_id.item() == 0:
                video = (
                    sio.Video(frame.video)
                    if isinstance(frame.video, str)
                    else sio.Video
                )
            lf, tracks = frame.to_slp(tracks, video=video)
            pred_slp.append(lf)
    pred_slp = sio.Labels(pred_slp)
    print(pred_slp)
    return pred_slp


@hydra.main(config_path=None, config_name=None, version_base=None)
def run(cfg: DictConfig) -> dict[int, sio.Labels]:
    """Run inference based on config file.

    Args:
        cfg: A dictconfig loaded from hydra containing checkpoint path and data
    """
    pred_cfg = Config(cfg)

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

        if pred_cfg.set_hparams(hparams):
            logger.info("Updated the following hparams to the following values")
            logger.info(hparams)
    else:
        hparams = {}

    checkpoint = pred_cfg.cfg.ckpt_path

    logging.getLogger().setLevel(level=cfg.get("log_level", "INFO").upper())

    model = GTRRunner.load_from_checkpoint(checkpoint, strict=False)
    tracker_cfg = pred_cfg.get_tracker_cfg()

    model.tracker_cfg = tracker_cfg
    model.tracker = Tracker(**model.tracker_cfg)

    logger.info(f"Using the following tracker:")
    logger.info(model.tracker)

    labels_files, vid_files = pred_cfg.get_data_paths(pred_cfg.cfg.dataset.test_dataset)
    trainer = pred_cfg.get_trainer()
    outdir = pred_cfg.cfg.outdir if "outdir" in pred_cfg.cfg else "./results"
    os.makedirs(outdir, exist_ok=True)

    for label_file, vid_file in zip(labels_files, vid_files):
        logger.info(f"Tracking {label_file} - {vid_file}...")
        dataset = pred_cfg.get_dataset(
            label_files=[label_file], vid_files=[vid_file], mode="test"
        )
        dataloader = pred_cfg.get_dataloader(dataset, mode="test")
        preds = track(model, trainer, dataloader)
        outpath = os.path.join(
            outdir, f"{Path(label_file).stem}.dreem_inference.{get_timestamp()}.slp"
        )

        preds.save(outpath)

    return preds


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py --config-dir=./configs --config-name=inference

    # override with params config:
    # python train.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml

    # override with params config, and specific params:
    # python train.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml dataset.train_dataset.padding=10
    run()
