"""Module containing model helper functions."""

from typing import Iterable
from pytorch_lightning import loggers
import torch


def get_boxes(instances: list["dreem.io.Instance"]) -> torch.Tensor:
    """Extract the bounding boxes from the input list of instances.

    Args:
        instances: List of Instance objects.

    Returns:
        An (n_instances, n_points, 4) float tensor containing the bounding boxes
        normalized by the height and width of the image
    """
    boxes = []
    for i, instance in enumerate(instances):
        _, h, w = instance.frame.img_shape.flatten()
        bbox = instance.bbox.clone()
        bbox[:, :, [0, 2]] /= w
        bbox[:, :, [1, 3]] /= h
        boxes.append(bbox)

    boxes = torch.cat(boxes, dim=0)  # N, n_anchors, 4

    return boxes


def get_times(
    ref_instances: list["dreem.io.Instance"],
    query_instances: list["dreem.io.Instance"] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the time indices of each instance relative to the window length.

    Args:
        ref_instances: Set of instances to query against
        query_instances: Set of query instances to look up using decoder.

    Returns:
        Tuple of Corresponding frame indices eg [0, 0, 1, 1, ..., T, T] for ref and query instances.
    """
    ref_inds = torch.tensor(
        [instance.frame.frame_id.item() for instance in ref_instances],
        device=ref_instances[0].device,
    )

    if query_instances is not None:
        query_inds = torch.tensor(
            [instance.frame.frame_id.item() for instance in query_instances],
            device=ref_inds.device,
        )
    else:
        query_inds = torch.tensor([], device=ref_inds.device)

    frame_inds = torch.concat([ref_inds, query_inds])
    window_length = len(frame_inds.unique())

    frame_idx_mapping = {frame_inds.unique()[i].item(): i for i in range(window_length)}
    ref_t = torch.tensor(
        [frame_idx_mapping[ind.item()] for ind in ref_inds], device=ref_inds.device
    )

    query_t = torch.tensor(
        [frame_idx_mapping[ind.item()] for ind in query_inds], device=ref_inds.device
    )

    return ref_t, query_t


def softmax_asso(asso_output: list[torch.Tensor]) -> list[torch.Tensor]:
    """Apply the softmax activation function on asso_output.

    Args:
        asso_output: Raw logits output of the tracking transformer. A list of
            torch tensors of shape (T, N_t, N_i) where:
                T: the length of the window
                N_t: number of instances in current/query frame (rightmost frame
                    of the window).
                N_i: number of detected instances in i-th frame of window.

    Returns:
        asso_output: Probabilities following softmax function, with same shape
            as input.
    """
    asso_active = []
    for asso in asso_output:
        asso = torch.cat([asso, asso.new_zeros((asso.shape[0], 1))], dim=1).softmax(
            dim=1
        )[:, :-1]
        asso_active.append(asso)

    return asso_active


def init_optimizer(params: Iterable, config: dict) -> torch.optim.Optimizer:
    """Initialize optimizer based on config parameters.

    Allows more flexibility in which optimizer to use

    Args:
        params: model parameters to be optimized
        config: optimizer hyperparameters including optimizer name

    Returns:
        optimizer: A torch.Optimizer with specified params
    """
    if config is None:
        config = {"name": "Adam"}
    optimizer = config.get("name", "Adam")
    optimizer_params = {
        param: val for param, val in config.items() if param.lower() != "name"
    }

    try:
        optimizer_class = getattr(torch.optim, optimizer)
    except AttributeError:
        if optimizer_class is None:
            print(
                f"Couldn't instantiate {optimizer} as given. Trying with capitalization"
            )
            optimizer_class = getattr(torch.optim, optimizer.lower().capitalize())
        if optimizer_class is None:
            print(
                f"Couldnt instantiate {optimizer} with capitalization, Final attempt with all caps"
            )
            optimizer_class = getattr(torch.optim, optimizer.upper(), None)

    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer type: {optimizer}")

    return optimizer_class(params, **optimizer_params)


def init_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> torch.optim.lr_scheduler.LRScheduler:
    """Initialize scheduler based on config parameters.

    Allows more flexibility in choosing which scheduler to use.

    Args:
        optimizer: optimizer for which to adjust lr
        config: lr scheduler hyperparameters including scheduler name

    Returns:
        scheduler: A scheduler with specified params
    """
    if config is None:
        return None
    scheduler = config.get("name")

    if scheduler is None:
        scheduler = "ReduceLROnPlateau"

    scheduler_params = {
        param: val for param, val in config.items() if param.lower() != "name"
    }

    try:
        # if a list is provided, apply each one sequentially
        if isinstance(scheduler, list):
            schedulers = []
            milestones = scheduler_params.get("milestones", None)
            for ix, s in enumerate(scheduler):
                params = scheduler_params[str(ix)]
                schedulers.append(
                    getattr(torch.optim.lr_scheduler, s)(optimizer, **params)
                )
            scheduler_class = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers, milestones
            )
            return scheduler_class

        else:
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler)
    except AttributeError:
        if scheduler_class is None:
            print(
                f"Couldn't instantiate {scheduler} as given. Trying with capitalization"
            )
            scheduler_class = getattr(
                torch.optim.lr_scheduler, scheduler.lower().capitalize()
            )
        if scheduler_class is None:
            print(
                f"Couldnt instantiate {scheduler} with capitalization, Final attempt with all caps"
            )
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler.upper(), None)

    if scheduler_class is None:
        raise ValueError(f"Unsupported optimizer type: {scheduler}")

    return scheduler_class(optimizer, **scheduler_params)


def init_logger(logger_params: dict, config: dict | None = None) -> loggers.Logger:
    """Initialize logger based on config parameters.

    Allows more flexibility in choosing which logger to use.

    Args:
        logger_params: logger hyperparameters
        config: rest of hyperparameters to log (mostly used for WandB)

    Returns:
        logger: A logger with specified params (or None).
    """
    logger_type = logger_params.pop("logger_type", None)

    valid_loggers = [
        "CSVLogger",
        "TensorBoardLogger",
        "WandbLogger",
    ]

    if logger_type in valid_loggers:
        logger_class = getattr(loggers, logger_type)
        if logger_class == loggers.WandbLogger:
            try:
                return logger_class(config=config, **logger_params)
            except Exception as e:
                print(e, logger_type)
        else:
            try:
                return logger_class(**logger_params)
            except Exception as e:
                print(e, logger_type)
    else:
        print(
            f"{logger_type} not one of {valid_loggers} or set to None, skipping logging"
        )
        return None


def get_device() -> str:
    """Utility function to get available device.

    Returns:
        str: The available device (one of 'cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device
