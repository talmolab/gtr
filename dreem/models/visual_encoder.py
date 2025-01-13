"""Module for different visual feature extractors."""

from typing import Any, Type, Dict
import torch
import torchvision
import timm
import torch.nn.functional as F
import skimage.measure as measure
from skimage.feature import local_binary_pattern
import numpy as np

# import timm

# todo: do we want to make timm a dependency?
# todo: add named tensor support

ENCODER_REGISTRY: Dict[str, Type[torch.nn.Module]] = {}


class VisualEncoder(torch.nn.Module):
    """Class wrapping around a visual feature extractor backbone.

    Currently CNN only.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        d_model: int = 512,
        in_chans: int = 3,
        backend: int = "timm",
        **kwargs: Any | None,
    ):
        """Initialize Visual Encoder.

        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            d_model (int): Output embedding dimension.
            in_chans: the number of input channels of the image.
            backend: Which model backend to use. One of {"timm", "torchvision"}
            kwargs: see `timm.create_model` and `torchvision.models.resnetX` for kwargs.
        """
        super().__init__()

        self.model_name = model_name.lower()
        self.d_model = d_model
        self.backend = backend
        if in_chans == 1:
            self.in_chans = 3
        else:
            self.in_chans = in_chans

        self.feature_extractor = self.select_feature_extractor(
            model_name=self.model_name,
            in_chans=self.in_chans,
            backend=self.backend,
            **kwargs,
        )

        self.out_layer = torch.nn.Linear(
            self.encoder_dim(self.feature_extractor), self.d_model
        )

    def select_feature_extractor(
        self, model_name: str, in_chans: int, backend: str, **kwargs: Any
    ) -> torch.nn.Module:
        """Select the appropriate feature extractor based on config.

        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            in_chans: the number of input channels of the image.
            backend: Which model backend to use. One of {"timm", "torchvision"}
            kwargs: see `timm.create_model` and `torchvision.models.resnetX` for kwargs.

        Returns:
            a CNN encoder based on the config and backend selected.
        """
        if "timm" in backend.lower():
            feature_extractor = timm.create_model(
                model_name=self.model_name,
                in_chans=self.in_chans,
                num_classes=0,
                **kwargs,
            )
        elif "torch" in backend.lower():
            if model_name.lower() == "resnet18":
                feature_extractor = torchvision.models.resnet18(**kwargs)

            elif model_name.lower() == "resnet50":
                feature_extractor = torchvision.models.resnet50(**kwargs)

            else:
                raise ValueError(
                    f"Only `[resnet18, resnet50]` are available when backend is {backend}. Found {model_name}"
                )
            feature_extractor = torch.nn.Sequential(
                *list(feature_extractor.children())[:-1]
            )
            input_layer = feature_extractor[0]
            if in_chans != 3:
                feature_extractor[0] = torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=input_layer.out_channels,
                    kernel_size=input_layer.kernel_size,
                    stride=input_layer.stride,
                    padding=input_layer.padding,
                    dilation=input_layer.dilation,
                    groups=input_layer.groups,
                    bias=input_layer.bias,
                    padding_mode=input_layer.padding_mode,
                )

        else:
            raise ValueError(
                f"Only ['timm', 'torch'] backends are available! Found {backend}."
            )
        return feature_extractor

    def encoder_dim(self, model: torch.nn.Module) -> int:
        """Compute dummy forward pass of encoder model and get embedding dimension.

        Args:
            model: a vision encoder model.

        Returns:
            The embedding dimension size.
        """
        _ = model.eval()
        dummy_output = model(torch.randn(1, self.in_chans, 224, 224)).squeeze()
        _ = model.train()  # to be safe
        return dummy_output.shape[-1]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector.

        Args:
            img: Input image tensor of shape (B, C, H, W).

        Returns:
            feats: Normalized output tensor of shape (B, d_model).
        """
        # If grayscale, tile the image to 3 channels.
        if img.shape[1] == 1:
            img = img.repeat([1, 3, 1, 1])  # (B, nc=3, H, W)

        b, c, h, w = img.shape

        if c != self.in_chans:
            raise ValueError(
                f"""Found {c} channels in image but model was configured for {self.in_chans} channels! \n
                    Hint: have you set the number of anchors in your dataset > 1? \n
                    If so, make sure to set `in_chans=3 * n_anchors`"""
            )
        feats = self.feature_extractor(
            img
        )  # (B, out_dim, 1, 1) if using resnet18 backbone.

        # Reshape feature vectors
        feats = feats.reshape([img.shape[0], -1])  # (B, out_dim)
        # Map feature vectors to output dimension using linear layer.
        feats = self.out_layer(feats)  # (B, d_model)
        # Normalize output feature vectors.
        feats = F.normalize(feats)  # (B, d_model)
        return feats


class DescriptorVisualEncoder(torch.nn.Module):
    """Visual Encoder based on image descriptors."""

    def __init__(self, use_hu_moments: bool = False, **kwargs):
        """Initialize Descriptor Visual Encoder.

        Args:
            use_hu_moments: Whether to use Hu moments.
        """
        super().__init__()
        self.use_hu_moments = use_hu_moments

    def compute_hu_moments(self, img):
        """Compute Hu moments."""
        mu = measure.moments_central(img)
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)
        # log transform hu moments for scale differences; switched off; numerically unstable
        # hu_log = -np.sign(hu) * np.log(np.abs(hu))

        return hu

    def compute_inertia_tensor(self, img):
        """Compute inertia tensor."""
        return measure.inertia_tensor(img)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector."""
        descriptors = []

        for im in img:
            im = im[0].cpu().numpy()

            inertia_tensor = self.compute_inertia_tensor(im)
            mean_intensity = im.mean()
            if self.use_hu_moments:
                hu_moments = self.compute_hu_moments(im)

            # Flatten inertia tensor
            inertia_tensor_flat = inertia_tensor.flatten()

            # Combine all features into a single descriptor
            descriptor = np.concatenate(
                [
                    inertia_tensor_flat,
                    [mean_intensity],
                    hu_moments if self.use_hu_moments else [],
                ]
            )

            descriptors.append(torch.tensor(descriptor, dtype=torch.float32))

        return torch.stack(descriptors)


def register_encoder(encoder_type: str, encoder_class: Type[torch.nn.Module]):
    """Register a new encoder type."""
    if not issubclass(encoder_class, torch.nn.Module):
        raise ValueError(f"{encoder_class} must be a subclass of torch.nn.Module")
    ENCODER_REGISTRY[encoder_type] = encoder_class


def create_visual_encoder(d_model: int, **encoder_cfg) -> torch.nn.Module:
    """Create a visual encoder based on the specified type."""
    register_encoder("resnet", VisualEncoder)
    register_encoder("descriptor", DescriptorVisualEncoder)
    # register any custom encoders here

    # compatibility with configs that don't specify encoder_type; default to resnet
    if not encoder_cfg or "encoder_type" not in encoder_cfg:
        encoder_type = "resnet"
        return ENCODER_REGISTRY[encoder_type](d_model=d_model, **encoder_cfg)
    else:
        encoder_type = encoder_cfg.pop("encoder_type")

    if encoder_type in ENCODER_REGISTRY:
        # choose the relevant encoder configs based on the encoder_type
        configs = encoder_cfg[encoder_type]
        return ENCODER_REGISTRY[encoder_type](d_model=d_model, **configs)
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Please use one of {list(ENCODER_REGISTRY.keys())}"
        )
