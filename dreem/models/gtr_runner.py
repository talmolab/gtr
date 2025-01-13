"""Module containing training, validation and inference logic."""

import torch
import gc
import logging
import pandas as pd
import h5py
from dreem.inference import Tracker
from dreem.inference import metrics
from dreem.models import GlobalTrackingTransformer
from dreem.training.losses import AssoLoss
from dreem.models.model_utils import init_optimizer, init_scheduler
from pytorch_lightning import LightningModule


logger = logging.getLogger("dreem.models")


class GTRRunner(LightningModule):
    """A lightning wrapper around GTR model.

    Used for training, validation and inference.
    """

    DEFAULT_METRICS = {
        "train": [],
        "val": ["num_switches"],
        "test": ["num_switches"],
    }
    DEFAULT_TRACKING = {
        "train": False,
        "val": True,
        "test": True,
    }
    DEFAULT_SAVE = {"train": False, "val": False, "test": False}

    def __init__(
        self,
        model_cfg: dict | None = None,
        tracker_cfg: dict | None = None,
        loss_cfg: dict | None = None,
        optimizer_cfg: dict | None = None,
        scheduler_cfg: dict | None = None,
        metrics: dict[str, list[str]] | None = None,
        persistent_tracking: dict[str, bool] | None = None,
        test_save_path: str = "./test_results.h5",
    ):
        """Initialize a lightning module for GTR.

        Args:
            model_cfg: hyperparameters for GlobalTrackingTransformer
            tracker_cfg: The parameters used for the tracker post-processing
            loss_cfg: hyperparameters for AssoLoss
            optimizer_cfg: hyper parameters used for optimizer.
                       Only used to overwrite `configure_optimizer`
            scheduler_cfg: hyperparameters for lr_scheduler used to overwrite `configure_optimizer
            metrics: a dict containing the metrics to be computed during train, val, and test.
            persistent_tracking: a dict containing whether to use persistent tracking during train, val and test inference.
            test_save_path: path to an .h5 file to save the test results to
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg if model_cfg else {}
        self.loss_cfg = loss_cfg if loss_cfg else {}
        self.tracker_cfg = tracker_cfg if tracker_cfg else {}

        self.model = GlobalTrackingTransformer(**self.model_cfg)
        self.loss = AssoLoss(**self.loss_cfg)
        self.tracker = Tracker(**self.tracker_cfg)

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.metrics = metrics if metrics is not None else self.DEFAULT_METRICS
        self.persistent_tracking = (
            persistent_tracking
            if persistent_tracking is not None
            else self.DEFAULT_TRACKING
        )
        self.test_results = {"metrics": [], "preds": [], "save_path": test_save_path}

    def forward(
        self,
        ref_instances: list["dreem.io.Instance"],
        query_instances: list["dreem.io.Instance"] | None = None,
    ) -> list["AssociationMatrix"]:
        """Execute forward pass of the lightning module.

        Args:
            ref_instances: a list of `Instance` objects containing crops and other data needed for transformer model
            query_instances: a list of `Instance` objects used as queries in the decoder. Mostly used for inference.

        Returns:
            An association matrix between objects
        """
        asso_preds = self.model(ref_instances, query_instances)
        return asso_preds

    def training_step(
        self, train_batch: list[list["dreem.io.Frame"]], batch_idx: int
    ) -> dict[str, float]:
        """Execute single training step for model.

        Args:
            train_batch: A single batch from the dataset which is a list of `Frame` objects
                        with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the train loss plus any other metrics specified
        """
        result = self._shared_eval_step(train_batch[0], mode="train")
        self.log_metrics(result, len(train_batch[0]), "train")

        return result

    def validation_step(
        self, val_batch: list[list["dreem.io.Frame"]], batch_idx: int
    ) -> dict[str, float]:
        """Execute single val step for model.

        Args:
            val_batch: A single batch from the dataset which is a list of `Frame` objects
                        with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the val loss plus any other metrics specified
        """
        result = self._shared_eval_step(val_batch[0], mode="val")
        self.log_metrics(result, len(val_batch[0]), "val")

        return result

    def test_step(
        self, test_batch: list[list["dreem.io.Frame"]], batch_idx: int
    ) -> dict[str, float]:
        """Execute single test step for model.

        Args:
            test_batch: A single batch from the dataset which is a list of `Frame` objects
                        with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the val loss plus any other metrics specified
        """
        result = self._shared_eval_step(test_batch[0], mode="test")
        self.log_metrics(result, len(test_batch[0]), "test")

        return result

    def predict_step(
        self, batch: list[list["dreem.io.Frame"]], batch_idx: int
    ) -> list["dreem.io.Frame"]:
        """Run inference for model.

        Computes association + assignment.

        Args:
            batch: A single batch from the dataset which is a list of `Frame` objects
                    with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            A list of dicts where each dict is a frame containing the predicted track ids
        """
        frames_pred = self.tracker(self.model, batch[0])
        return frames_pred

    def _shared_eval_step(
        self, frames: list["dreem.io.Frame"], mode: str
    ) -> dict[str, float]:
        """Run evaluation used by train, test, and val steps.

        Args:
            frames: A list of dicts where each dict is a frame containing gt data
            mode: which metrics to compute and whether to use persistent tracking or not

        Returns:
            a dict containing the loss and any other metrics specified by `eval_metrics`
        """
        try:
            instances = [instance for frame in frames for instance in frame.instances]

            if len(instances) == 0:
                return None

            eval_metrics = self.metrics[mode]
            persistent_tracking = self.persistent_tracking[mode]

            logits = self(instances)
            logits = [asso.matrix for asso in logits]
            loss = self.loss(logits, frames)

            return_metrics = {"loss": loss}
            if eval_metrics is not None and len(eval_metrics) > 0:
                self.tracker.persistent_tracking = persistent_tracking

                frames_pred = self.tracker(self.model, frames)

                frames_mm = metrics.to_track_eval(frames_pred)
                clearmot = metrics.get_pymotmetrics(frames_mm, eval_metrics)

                return_metrics.update(clearmot.to_dict())

                if mode == "test":
                    self.test_results["preds"].append(
                        [frame.to("cpu") for frame in frames_pred]
                    )
                    self.test_results["metrics"].append(return_metrics)
            return_metrics["batch_size"] = len(frames)
        except Exception as e:
            logger.exception(
                f"Failed on frame {frames[0].frame_id} of video {frames[0].video_id}"
            )
            logger.exception(e)
            raise (e)

        return return_metrics

    def configure_optimizers(self) -> dict:
        """Get optimizers and schedulers for training.

        Is overridden by config but defaults to Adam + ReduceLROnPlateau.

        Returns:
            an optimizer config dict containing the optimizer, scheduler, and scheduler params
        """
        # todo: init from config
        if self.optimizer_cfg is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        else:
            optimizer = init_optimizer(self.parameters(), self.optimizer_cfg)

        if self.scheduler_cfg is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", 0.5, 10
            )
        else:
            scheduler = init_scheduler(optimizer, self.scheduler_cfg)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def log_metrics(self, result: dict, batch_size: int, mode: str) -> None:
        """Log metrics computed during evaluation.

        Args:
            result: A dict containing metrics to be logged.
            batch_size: the size of the batch used to compute the metrics
            mode: One of {'train', 'test' or 'val'}. Used as prefix while logging.
        """
        if result:
            batch_size = result.pop("batch_size")
            for metric, val in result.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.log(f"{mode}_{metric}", val, batch_size=batch_size)

    def on_validation_epoch_end(self):
        """Execute hook for validation end.

        Currently, we simply clear the gpu cache and do garbage collection.
        """
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        """Execute hook for test end.

        Currently, we save results to an h5py file. and clear the predictions
        """
        fname = self.test_results["save_path"]
        test_results = {
            key: val for key, val in self.test_results.items() if key != "save_path"
        }
        metrics_dict = [
            {
                key: (
                    val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val
                )
                for key, val in metrics.items()
            }
            for metrics in test_results["metrics"]
        ]
        results_df = pd.DataFrame(metrics_dict)
        preds = test_results["preds"]

        with h5py.File(fname, "a") as results_file:
            for key in results_df.columns:
                avg_result = results_df[key].mean()
                results_file.attrs.create(key, avg_result)
            for i, (metrics, frames) in enumerate(zip(metrics_dict, preds)):
                vid_name = frames[0].vid_name.split("/")[-1]
                vid_group = results_file.require_group(vid_name)
                clip_group = vid_group.require_group(f"clip_{i}")
                for key, val in metrics.items():
                    clip_group.attrs.create(key, val)
                for frame in frames:
                    if metrics.get("num_switches", 0) > 0:
                        _ = frame.to_h5(
                            clip_group,
                            [
                                instance.gt_track_id.item()
                                for instance in frame.instances
                            ],
                            save={"crop": True, "features": True, "embeddings": True},
                        )
                    else:
                        _ = frame.to_h5(
                            clip_group,
                            [
                                instance.gt_track_id.item()
                                for instance in frame.instances
                            ],
                        )
        self.test_results = {"metrics": [], "preds": [], "save_path": fname}
