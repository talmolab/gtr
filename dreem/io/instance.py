"""Module containing data class for storing detections."""

import torch
import sleap_io as sio
import numpy as np
import attrs
import logging
import h5py
from numpy.typing import ArrayLike
from typing import Self, Any

logger = logging.getLogger("dreem.io")


def _to_tensor(data: float | ArrayLike) -> torch.Tensor:
    """Convert data to a torch.Tensor object.

    Args:
        data: Either a scalar quantity or arraylike object

    Returns:
        A torch Tensor containing data.
    """
    if data is None:
        return torch.tensor([])
    if isinstance(data, torch.Tensor):
        return data
    elif np.isscalar(data):
        return torch.tensor([data])
    else:
        return torch.tensor(data)


def _expand_to_rank(
    arr: np.ndarray | torch.Tensor, new_rank: int
) -> np.ndarray | torch.Tensor:
    """Expand n-dimensional array to appropriate dimensions by adding singleton dimensions to the front of the array.

    Args:
        arr: an n-dimensional array (either np.ndarray or torch.Tensor).

    Returns:
        The array expanded to the correct dimensions.
    """
    curr_rank = len(arr.shape)
    while curr_rank < new_rank:
        if isinstance(arr, np.ndarray):
            arr = np.expand_dims(arr, axis=0)
        elif isinstance(arr, torch.Tensor):
            arr = arr.unsqueeze(0)
        else:
            raise TypeError(
                f"`arr` must be either an np.ndarray or torch.Tensor but found {type(arr)}"
            )
        curr_rank = len(arr.shape)
    return arr


@attrs.define(eq=False)
class Instance:
    """Class representing a single instance to be tracked.

    Attributes:
        gt_track_id: Ground truth track id - only used for train/eval.
        pred_track_id: Predicted track id. Untracked instance is represented by -1.
        bbox: The bounding box coordinate of the instance. Defaults to an empty tensor.
        crop: The crop of the instance.
        centroid: the centroid around which the bbox was cropped.
        features: The reid features extracted from the CNN backbone used in the transformer.
        track_score: The track score output from the association matrix.
        point_scores: The point scores from sleap.
        instance_score: The instance scores from sleap.
        skeleton: The sleap skeleton used for the instance.
        pose: A dictionary containing the node name and corresponding point.
        device: String representation of the device the instance should be on.
    """

    _gt_track_id: int = attrs.field(
        alias="gt_track_id", default=-1, converter=_to_tensor
    )
    _pred_track_id: int = attrs.field(
        alias="pred_track_id", default=-1, converter=_to_tensor
    )
    _bbox: ArrayLike = attrs.field(alias="bbox", factory=list, converter=_to_tensor)
    _crop: ArrayLike = attrs.field(alias="crop", factory=list, converter=_to_tensor)
    _centroid: dict[str, ArrayLike] = attrs.field(alias="centroid", factory=dict)
    _features: ArrayLike = attrs.field(
        alias="features", factory=list, converter=_to_tensor
    )
    _embeddings: dict = attrs.field(alias="embeddings", factory=dict)
    _track_score: float = attrs.field(alias="track_score", default=-1.0)
    _instance_score: float = attrs.field(alias="instance_score", default=-1.0)
    _point_scores: ArrayLike | None = attrs.field(alias="point_scores", default=None)
    _skeleton: sio.Skeleton | None = attrs.field(alias="skeleton", default=None)
    _pose: dict[str, ArrayLike] = attrs.field(alias="pose", factory=dict)
    _device: str | torch.device | None = attrs.field(alias="device", default=None)
    _frame: "Frame" = None

    def __attrs_post_init__(self) -> None:
        """Handle dimensionality and more intricate default initializations post-init."""
        self.bbox = _expand_to_rank(self.bbox, 3)
        self.crop = _expand_to_rank(self.crop, 4)
        self.features = _expand_to_rank(self.features, 2)

        if self.skeleton is None:
            self.skeleton = sio.Skeleton(["centroid"])

        if self.bbox.shape[-1] == 0:
            self.bbox = torch.empty([1, 0, 4])

        if self.crop.shape[-1] == 0 and self.bbox.shape[1] != 0:
            y1, x1, y2, x2 = self.bbox.squeeze(dim=0).nanmean(dim=0)
            self.centroid = {"centroid": np.array([(x1 + x2) / 2, (y1 + y2) / 2])}

        if len(self.pose) == 0 and self.bbox.shape[1]:
            y1, x1, y2, x2 = self.bbox.squeeze(dim=0).mean(dim=0)
            self._pose = {"centroid": np.array([(x1 + x2) / 2, (y1 + y2) / 2])}

        if self.point_scores is None and len(self.pose) != 0:
            self._point_scores = np.zeros((len(self.pose), 2))

        self.to(self.device)

    def __repr__(self) -> str:
        """Return string representation of the Instance."""
        return (
            "Instance("
            f"gt_track_id={self._gt_track_id.item()}, "
            f"pred_track_id={self._pred_track_id.item()}, "
            f"bbox={self._bbox}, "
            f"centroid={self._centroid}, "
            f"crop={self._crop.shape}, "
            f"features={self._features.shape}, "
            f"device={self._device}"
            ")"
        )

    def to(self, map_location: str | torch.device) -> Self:
        """Move instance to different device or change dtype. (See `torch.to` for more info).

        Args:
            map_location: Either the device or dtype for the instance to be moved.

        Returns:
            self: reference to the instance moved to correct device/dtype.
        """
        if map_location is not None and map_location != "":
            self._gt_track_id = self._gt_track_id.to(map_location)
            self._pred_track_id = self._pred_track_id.to(map_location)
            self._bbox = self._bbox.to(map_location)
            self._crop = self._crop.to(map_location)
            self._features = self._features.to(map_location)
            if isinstance(map_location, (str, torch.device)):
                self.device = map_location

        return self

    @classmethod
    def from_slp(
        cls,
        slp_instance: sio.PredictedInstance | sio.Instance,
        bbox_size: int | tuple[int, int] = 64,
        crop: ArrayLike | None = None,
        device: str | None = None,
    ) -> Self:
        """Convert a slp instance to a dreem instance.

        Args:
            slp_instance: A `sleap_io.Instance` object representing a detection
            bbox_size: size of the pose-centered bbox to form.
            crop: The corresponding crop of the bbox
            device: which device to keep the instance on
        Returns:
            A dreem.Instance object with a pose-centered bbox and no crop.
        """
        try:
            track_id = int(slp_instance.track.name)
        except ValueError:
            track_id = int(
                "".join([str(ord(c)) for c in slp_instance.track.name])
            )  # better way to handle this?
        if isinstance(bbox_size, int):
            bbox_size = (bbox_size, bbox_size)

        track_score = -1.0
        point_scores = np.full(len(slp_instance.points), -1)
        instance_score = -1
        if isinstance(slp_instance, sio.PredictedInstance):
            track_score = slp_instance.tracking_score
            point_scores = slp_instance.numpy()[:, -1]
            instance_score = slp_instance.score

        centroid = np.nanmean(slp_instance.numpy(), axis=1)
        bbox = [
            centroid[1] - bbox_size[1],
            centroid[0] - bbox_size[0],
            centroid[1] + bbox_size[1],
            centroid[0] + bbox_size[0],
        ]
        return cls(
            gt_track_id=track_id,
            bbox=bbox,
            crop=crop,
            centroid={"centroid": centroid},
            track_score=track_score,
            point_scores=point_scores,
            instance_score=instance_score,
            skeleton=slp_instance.skeleton,
            pose={
                node.name: point.numpy() for node, point in slp_instance.points.items()
            },
            device=device,
        )

    def to_slp(
        self, track_lookup: dict[int, sio.Track] = {}
    ) -> tuple[sio.PredictedInstance, dict[int, sio.Track]]:
        """Convert instance to sleap_io.PredictedInstance object.

        Args:
            track_lookup: A track look up dictionary containing track_id:sio.Track.
        Returns: A sleap_io.PredictedInstance with necessary metadata
            and a track_lookup dictionary to persist tracks.
        """
        try:
            track_id = self.pred_track_id.item()
            if track_id not in track_lookup:
                track_lookup[track_id] = sio.Track(name=self.pred_track_id.item())

            track = track_lookup[track_id]

            return (
                sio.PredictedInstance.from_numpy(
                    points=np.array(list(self.pose.values())),
                    skeleton=self.skeleton,
                    point_scores=self.point_scores,
                    instance_score=self.instance_score,
                    tracking_score=self.track_score,
                    track=track,
                ),
                track_lookup,
            )
        except Exception as e:
            logger.exception(
                f"Pose: {np.array(list(self.pose.values())).shape}, Pose score shape {self.point_scores.shape}"
            )
            raise RuntimeError(f"Failed to convert to sio.PredictedInstance: {e}")

    def to_h5(
        self, frame_group: h5py.Group, label: Any = None, **kwargs: dict
    ) -> h5py.Group:
        """Convert instance to an h5 group".

        By default we always save:
            - the gt/pred track id
            - bbox
            - centroid
            - pose
            - instance/traj/points score
        Larger arrays (crops/features/embeddings) can be saved by passing as kwargs

        Args:
            frame_group: the h5py group representing the frame the instance appears on
            label: the name of the instance group that will be created
            **kwargs: additional key:value pairs to be saved as datasets.

        Returns:
            The h5 group representing this instance.
        """
        if label is None:
            if pred_track_id != -1:
                label = f"instance_{self.pred_track_id.item()}"
            else:
                label = f"instance_{self.gt_track_id.item()}"
        instance_group = frame_group.create_group(label)
        instance_group.attrs.create("gt_track_id", self.gt_track_id.item())
        instance_group.attrs.create("pred_track_id", self.pred_track_id.item())
        instance_group.attrs.create("track_score", self.track_score)
        instance_group.attrs.create("instance_score", self.instance_score)

        instance_group.create_dataset("bbox", data=self.bbox.cpu().numpy())

        pose_group = instance_group.create_group("pose")
        pose_group.create_dataset("points", data=np.array(list(self.pose.values())))
        pose_group.attrs.create("nodes", list(self.pose.keys()))
        pose_group.create_dataset("scores", data=self.point_scores)

        for key, value in kwargs.items():
            if "emb" in key:
                emb_group = instance_group.require_group("emb")
                emb_group.create_dataset(key, data=value)
            else:
                instance_group.create_dataset(key, data=value)

        return instance_group

    @property
    def device(self) -> str:
        """The device the instance is on.

        Returns:
            The str representation of the device the gpu is on.
        """
        return self._device

    @device.setter
    def device(self, device) -> None:
        """Set for the device property.

        Args:
            device: The str representation of the device.
        """
        self._device = device

    @property
    def gt_track_id(self) -> torch.Tensor:
        """The ground truth track id of the instance.

        Returns:
            A tensor containing the ground truth track id
        """
        return self._gt_track_id

    @gt_track_id.setter
    def gt_track_id(self, track: int):
        """Set the instance ground-truth track id.

        Args:
           track: An int representing the ground-truth track id.
        """
        if track is not None:
            self._gt_track_id = torch.tensor([track])
        else:
            self._gt_track_id = torch.tensor([])

    def has_gt_track_id(self) -> bool:
        """Determine if instance has a gt track assignment.

        Returns:
            True if the gt track id is set, otherwise False.
        """
        if self._gt_track_id.shape[0] == 0:
            return False
        else:
            return True

    @property
    def pred_track_id(self) -> torch.Tensor:
        """The track id predicted by the tracker using asso_output from model.

        Returns:
            A tensor containing the predicted track id.
        """
        return self._pred_track_id

    @pred_track_id.setter
    def pred_track_id(self, track: int) -> None:
        """Set predicted track id.

        Args:
            track: an int representing the predicted track id.
        """
        if track is not None:
            self._pred_track_id = torch.tensor([track])
        else:
            self._pred_track_id = torch.tensor([])

    def has_pred_track_id(self) -> bool:
        """Determine whether instance has predicted track id.

        Returns:
            True if instance has a pred track id, False otherwise.
        """
        if self._pred_track_id.item() == -1 or self._pred_track_id.shape[0] == 0:
            return False
        else:
            return True

    @property
    def bbox(self) -> torch.Tensor:
        """The bounding box coordinates of the instance in the original frame.

        Returns:
            A (1,4) tensor containing the bounding box coordinates.
        """
        return self._bbox

    @bbox.setter
    def bbox(self, bbox: ArrayLike) -> None:
        """Set the instance bounding box.

        Args:
            bbox: an arraylike object containing the bounding box coordinates.
        """
        if bbox is None or len(bbox) == 0:
            self._bbox = torch.empty((0, 4))
        else:
            if not isinstance(bbox, torch.Tensor):
                self._bbox = torch.tensor(bbox)
            else:
                self._bbox = bbox

        if self._bbox.shape[0] and len(self._bbox.shape) == 1:
            self._bbox = self._bbox.unsqueeze(0)
        if self._bbox.shape[1] and len(self._bbox.shape) == 2:
            self._bbox = self._bbox.unsqueeze(0)

    def has_bbox(self) -> bool:
        """Determine if the instance has a bbox.

        Returns:
            True if the instance has a bounding box, false otherwise.
        """
        if self._bbox.shape[1] == 0:
            return False
        else:
            return True

    @property
    def centroid(self) -> dict[str, ArrayLike]:
        """The centroid around which the crop was formed.

        Returns:
            A dict containing the anchor name and the x, y bbox midpoint.
        """
        return self._centroid

    @centroid.setter
    def centroid(self, centroid: dict[str, ArrayLike]) -> None:
        """Set the centroid of the instance.

        Args:
            centroid: A dict containing the anchor name and points.
        """
        self._centroid = centroid

    @property
    def anchor(self) -> list[str]:
        """The anchor node name around which the crop was formed.

        Returns:
            the list of anchors around which each crop was formed
            the list of anchors around which each crop was formed
        """
        if self.centroid:
            return list(self.centroid.keys())
        return ""

    @property
    def crop(self) -> torch.Tensor:
        """The crop of the instance.

        Returns:
            A (1, c, h , w) tensor containing the cropped image centered around the instance.
        """
        return self._crop

    @crop.setter
    def crop(self, crop: ArrayLike) -> None:
        """Set the crop of the instance.

        Args:
            crop: an arraylike object containing the cropped image of the centered instance.
        """
        if crop is None or len(crop) == 0:
            self._crop = torch.tensor([])
        else:
            if not isinstance(crop, torch.Tensor):
                self._crop = torch.tensor(crop)
            else:
                self._crop = crop

        if len(self._crop.shape) == 2:
            self._crop = self._crop.unsqueeze(0)
        if len(self._crop.shape) == 3:
            self._crop = self._crop.unsqueeze(0)

    def has_crop(self) -> bool:
        """Determine if the instance has a crop.

        Returns:
            True if the instance has an image otherwise False.
        """
        if self._crop.shape[-1] == 0:
            return False
        else:
            return True

    @property
    def features(self) -> torch.Tensor:
        """Re-ID feature vector from backbone model to be used as input to transformer.

        Returns:
            a (1, d) tensor containing the reid feature vector.
        """
        return self._features

    @features.setter
    def features(self, features: ArrayLike) -> None:
        """Set the reid feature vector of the instance.

        Args:
            features: a (1,d) array like object containing the reid features for the instance.
        """
        if features is None or len(features) == 0:
            self._features = torch.tensor([])

        elif not isinstance(features, torch.Tensor):
            self._features = torch.tensor(features)
        else:
            self._features = features

        if self._features.shape[0] and len(self._features.shape) == 1:
            self._features = self._features.unsqueeze(0)

    def has_features(self) -> bool:
        """Determine if the instance has computed reid features.

        Returns:
            True if the instance has reid features, False otherwise.
        """
        if self._features.shape[-1] == 0:
            return False
        else:
            return True

    def has_embedding(self, emb_type: str | None = None) -> bool:
        """Determine if the instance has embedding type requested.

        Args:
            emb_type: The key to check in the embedding dictionary.

        Returns:
            True if `emb_type` in embedding_dict else false
        """
        return emb_type in self._embeddings

    def get_embedding(
        self, emb_type: str = "all"
    ) -> dict[str, torch.Tensor] | torch.Tensor | None:
        """Retrieve instance's spatial/temporal embedding.

        Args:
            emb_type: The string key of the embedding to retrieve. Should be "pos", "temp"

        Returns:
            * A torch tensor representing the spatial/temporal location of the instance.
            * None if the embedding is not stored
        """
        if emb_type.lower() == "all":
            return self._embeddings
        else:
            try:
                return self._embeddings[emb_type]
            except KeyError:
                logger.exception(
                    f"{emb_type} not saved! Only {list(self._embeddings.keys())} are available"
                )
        return None

    def add_embedding(self, emb_type: str, embedding: torch.Tensor) -> None:
        """Save embedding to instance embedding dictionary.

        Args:
            emb_type: Key/embedding type to be saved to dictionary
            embedding: The actual torch tensor embedding.
        """
        if (
            type(embedding) != dict
        ):  # for embedding agg method "average", input is array
            # for method stack and concatenate, input is dict
            embedding = _expand_to_rank(embedding, 2)
        self._embeddings[emb_type] = embedding

    @property
    def frame(self) -> "Frame":
        """Get the frame the instance belongs to.

        Returns:
            The back reference to the `Frame` that this `Instance` belongs to.
        """
        return self._frame

    @frame.setter
    def frame(self, frame: "Frame") -> None:
        """Set the back reference to the `Frame` that this `Instance` belongs to.

        This field is set when instances are added to `Frame` object.

        Args:
            frame: A `Frame` object containing the metadata for the frame that the instance belongs to
        """
        self._frame = frame

    @property
    def pose(self) -> dict[str, ArrayLike]:
        """Get the pose of the instance.

        Returns:
            A dictionary containing the node and corresponding x,y points
        """
        return self._pose

    @pose.setter
    def pose(self, pose: dict[str, ArrayLike]) -> None:
        """Set the pose of the instance.

        Args:
            pose: A nodes x 2 array containing the pose coordinates.
        """
        if pose is not None:
            self._pose = pose

        elif self.bbox.shape[0]:
            y1, x1, y2, x2 = self.bbox.squeeze()
            self._pose = {"centroid": np.array([(x1 + x2) / 2, (y1 + y2) / 2])}

        else:
            self._pose = {}

    def has_pose(self) -> bool:
        """Check if the instance has a pose.

        Returns True if the instance has a pose.
        """
        if len(self.pose):
            return True
        return False

    @property
    def shown_pose(self) -> dict[str, ArrayLike]:
        """Get the pose with shown nodes only.

        Returns: A dictionary filtered by nodes that are shown (points are not nan).
        """
        pose = self.pose
        return {node: point for node, point in pose.items() if not np.isna(point).any()}

    @property
    def skeleton(self) -> sio.Skeleton:
        """Get the skeleton associated with the instance.

        Returns: The sio.Skeleton associated with the instance.
        """
        return self._skeleton

    @skeleton.setter
    def skeleton(self, skeleton: sio.Skeleton) -> None:
        """Set the skeleton associated with the instance.

        Args:
            skeleton: The sio.Skeleton associated with the instance.
        """
        self._skeleton = skeleton

    @property
    def point_scores(self) -> ArrayLike:
        """Get the point scores associated with the pose prediction.

        Returns: a vector of shape n containing the point scores outputed from sleap associated with pose predictions.
        """
        return self._point_scores

    @point_scores.setter
    def point_scores(self, point_scores: ArrayLike) -> None:
        """Set the point scores associated with the pose prediction.

        Args:
            point_scores: a vector of shape n containing the point scores
            outputted from sleap associated with pose predictions.
        """
        self._point_scores = point_scores

    @property
    def instance_score(self) -> float:
        """Get the pose prediction score associated with the instance.

        Returns: a float from 0-1 representing an instance_score.
        """
        return self._instance_score

    @instance_score.setter
    def instance_score(self, instance_score: float) -> None:
        """Set the pose prediction score associated with the instance.

        Args:
            instance_score: a float from 0-1 representing an instance_score.
        """
        self._instance_score = instance_score

    @property
    def track_score(self) -> float:
        """Get the track_score of the instance.

        Returns: A float from 0-1 representing the output used in the tracker for assignment.
        """
        return self._track_score

    @track_score.setter
    def track_score(self, track_score: float) -> None:
        """Set the track_score of the instance.

        Args:
            track_score: A float from 0-1 representing the output used in the tracker for assignment.
        """
        self._track_score = track_score
