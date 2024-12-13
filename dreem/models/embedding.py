"""Module containing different position and temporal embeddings."""

import math
import torch
import logging
from torch import nn, Tensor
from typing import Optional
from dreem.models.mlp import MLP

logger = logging.getLogger("dreem.models")
# todo: add named tensors, clean variable names


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        # max_seq_len: int,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        # self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)

    def build_rope_cache(self, max_seq_len: int) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`

        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # create the lookup array based on how many instances there are
        # max(101, seq_len) is for positional vs temporal; pos can only have idx up to
        # 100 since it's a fraction of [0,1]*100. temp is from [0, clip_len]; since clip_len
        # not available, we use the last value in the indexing array since this will be the
        # last possible frame that we would need to index since no instances in a frame after that
        if input_pos.dim() <= 1: input_pos = input_pos.unsqueeze(0)
        self.build_rope_cache(max(101, input_pos[:, -1].max() + 1))  # registers cache
        self.cache = self.cache.to(input_pos.device)
        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        return rope_cache


class Embedding(torch.nn.Module):
    """Class that wraps around different embedding types.
    Creates embedding array and transforms the input data
    Used for both learned and fixed embeddings.
    """

    EMB_TYPES = {
        "temp": {},
        "pos": {"over_boxes"},
        "off": {},
        None: {},
    }  # dict of valid args:keyword params
    EMB_MODES = {
        "fixed": {"temperature", "scale", "normalize"},
        "learned": {"emb_num"},
        "rope": {"embedding_agg_method"},
        "off": {},
    }  # dict of valid args:keyword params

    def __init__(
        self,
        emb_type: str,
        mode: str,
        features: int,
        n_points: int = 1,
        emb_num: int = 16,
        over_boxes: bool = True,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
        mlp_cfg: dict | None = None,
        embedding_agg_method: str = "average",
    ):
        """Initialize embeddings.

        Args:
            emb_type: The type of embedding to compute. Must be one of `{"temp", "pos", "off"}`
            mode: The mode or function used to map positions to vector embeddings.
                  Must be one of `{"fixed", "learned", "off"}`
            features: The embedding dimensions. Must match the dimension of the
                      input vectors for the transformer model.
            n_points: the number of points that will be embedded.
            emb_num: the number of embeddings in the `self.lookup` table (Only used in learned embeddings).
            over_boxes: Whether to compute the position embedding for each bbox coordinate (y1x1y2x2) or the centroid + bbox size (yxwh).
            temperature: the temperature constant to be used when computing the sinusoidal position embedding
            normalize: whether or not to normalize the positions (Only used in fixed embeddings).
            scale: factor by which to scale the positions after normalizing (Only used in fixed embeddings).
            mlp_cfg: A dictionary of mlp hyperparameters for projecting embedding to correct space.
                    Example: {"hidden_dims": 256, "num_layers":3, "dropout": 0.3}
        """

        super().__init__()

        self.emb_type = emb_type
        self.mode = mode
        self.embedding_agg_method = embedding_agg_method
        self.features = features
        self.emb_num = emb_num
        self.over_boxes = over_boxes
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.n_points = n_points

        self._check_init_args(emb_type, mode)

        if self.normalize and self.scale is None:
            self.scale = 2 * math.pi

        if self.emb_type == "pos" and mlp_cfg is not None and mlp_cfg["num_layers"] > 0:
            if self.mode == "fixed":
                self.mlp = MLP(
                    input_dim=n_points * self.features,
                    output_dim=self.features,
                    **mlp_cfg,
                )
            else:
                in_dim = (self.features // (4 * n_points)) * (4 * n_points)
                self.mlp = MLP(
                    input_dim=in_dim,
                    output_dim=self.features,
                    **mlp_cfg,
                )
        else:
            self.mlp = torch.nn.Identity()

        self._emb_func = lambda seq, x: torch.zeros(
            (seq.shape[0], self.features), dtype=seq.dtype, device=seq.device
        )  # turn off embedding by returning zeros

        self.lookup = None

        if self.mode == "learned":
            if self.emb_type == "pos":
                self.lookup = torch.nn.Embedding(
                    self.emb_num * 4 * self.n_points, self.features // (4 * n_points)
                )
                self._emb_func = self._learned_pos_embedding
            elif self.emb_type == "temp":
                self.lookup = torch.nn.Embedding(self.emb_num, self.features)
                self._emb_func = self._learned_temp_embedding

        elif self.mode == "fixed":
            if self.emb_type == "pos":
                if self.embedding_agg_method == "average":
                    self._emb_func = self._sine_box_embedding
                else:  # if using stacked/concatenated agg method
                    self._emb_func = self._sine_pos_embedding
            elif self.emb_type == "temp":
                self._emb_func = self._sine_temp_embedding

        elif self.mode == "rope":
            # pos/temp embeddings processed the same way with different embedding array inputs
            self._emb_func = self._rope_embedding
            # create instance so embedding lookup array is created only once
            self.rope_instance = RotaryPositionalEmbeddings(self.features)

    def _check_init_args(self, emb_type: str, mode: str):
        """Check whether the correct arguments were passed to initialization.

        Args:
            emb_type: The type of embedding to compute. Must be one of `{"temp", "pos", ""}`
            mode: The mode or function used to map positions to vector embeddings.
                Must be one of `{"fixed", "learned"}`

        Raises:
            ValueError:
              * if the incorrect `emb_type` or `mode` string are passed
            NotImplementedError: if `emb_type` is `temp` and `mode` is `fixed`.
        """
        if emb_type.lower() not in self.EMB_TYPES:
            raise ValueError(
                f"Embedding `emb_type` must be one of {self.EMB_TYPES} not {emb_type}"
            )

        if mode.lower() not in self.EMB_MODES:
            raise ValueError(
                f"Embedding `mode` must be one of {self.EMB_MODES} not {mode}"
            )

        if mode.lower() == "rope" and self.embedding_agg_method == "average":
            raise ValueError(
                f"Cannot use aggregation method 'average' for rope embedding; must use 'stack' or 'concatenate'"
            )

    def _transform(self, x, emb):
        """Routes to the relevant embedding function to transform the input queries

        Args:
             x: Input queries of shape (batch_size, N, embed_dim)
             emb: Embedding array to apply to data; can be (N, embed_dim) or
             (batch_size, n_query, num_heads, embed_dim // 2, 2) if using RoPE
        """
        if self._emb_func == self._rope_embedding:
            return self._apply_rope(x, emb)
        else:
            return self._apply_additive_embeddings(x, emb)

    def _apply_rope(self, x, emb):
        """Applies Rotary Positional Embedding to input queries

        Args:
            x: Input queries of shape (batch_size, n_query, embed_dim)
            emb: Rotation matrix of shape (batch_size, n_query, num_heads, embed_dim // 2, 2)

        Returns:
            Tensor of input queries transformed by RoPE
        """

        xout = torch.unsqueeze(x, 2)
        # input needs shape [batch_size, n_query, num_heads, embed_dim // 2, 2]
        xout = xout.float().reshape(*xout.shape[:-1], -1, 2)
        # apply RoPE to each query token
        xout = torch.stack(
            [
                xout[..., 0] * emb[..., 0] - xout[..., 1] * emb[..., 1],
                xout[..., 1] * emb[..., 0] + xout[..., 0] * emb[..., 1],
            ],
            -1,
        )
        # output has shape [batch_size, n_query, num_heads, embed_dim]
        xout = xout.flatten(3).squeeze(2)

        return xout

    def _apply_additive_embeddings(self, x, emb):
        """Applies additive embeddings to input queries

        Args:
            x: Input tensor of shape (batch_size, N, embed_dim)
            emb: Embedding array of shape (N, embed_dim)

        Returns:
            Tensor: Input queries with embeddings added - shape (batch_size, N, embed_dim)
        """
        _emb = emb.unsqueeze(0)
        return x + _emb

    def forward(self, x, seq_positions: torch.Tensor) -> torch.Tensor:
        """Get the sequence positional embeddings.

        Args:
            seq_positions:
                * An (N,) tensor where seq_positions[i] represents the temporal position of instance_i in the sequence.
                * An (N, n_anchors x 4) tensor where seq_positions[i, j, :] represents the [y1, x1, y2, x2] spatial locations of jth point of instance_i in the sequence.
            x: Input data of shape ((batch_size, N, embed_dim))

        Returns:
            - Tensor: input queries transformed by embedding
            - An `N` x `self.features` tensor representing the corresponding spatial or temporal embedding.
        """

        # create embedding array; either rotation matrix of shape
        # (batch_size, n_query, num_heads, embed_dim // 2, 2),
        # or (N, embed_dim) array
        emb = self._emb_func(seq_positions, x.size())
        # transform the input data with the embedding
        xout = self._transform(x, emb)

        return xout, emb

    def _torch_int_div(
        self, tensor1: torch.Tensor, tensor2: torch.Tensor
    ) -> torch.Tensor:
        """Perform integer division of two tensors.

        Args:
            tensor1: dividend tensor.
            tensor2: divisor tensor.

        Returns:
            torch.Tensor, resulting tensor.
        """
        return torch.div(tensor1, tensor2, rounding_mode="floor")

    def _rope_embedding(
        self, seq_positions: torch.Tensor, input_shape: torch.Size
    ) -> torch.Tensor:
        """Computes the rotation matrix to apply RoPE to input queries
        Args:
            seq_positions: Pos array of shape (embed_dim,) used to compute rotational embedding
            input_shape: Shape of the input queries; needed for rope
        Returns:
            Tensor: (N, embed_dim) rotation matrix
        """
        # create dummy input of shape (num_batches, num_instances, num_attn_heads, embed_dim)
        # use num_heads=1 for compatibility with torch ROPE
        x_rope = torch.rand(input_shape).unsqueeze(2)
        # infer whether it is a positional or temporal embedding
        is_pos_emb = 1 if seq_positions.max() < 1 else 0
        # if it is positional, scale seq_positions since these are fractions
        # in [0,1] and we need int indexes for embedding lookup
        seq_positions = seq_positions * 100 if is_pos_emb else seq_positions
        seq_positions = seq_positions.unsqueeze(0).int()
        # RoPE module takes in dimension, num_queries as input to calculate rotation matrix
        rot_mat = self.rope_instance(x_rope, seq_positions)

        return rot_mat

    def _sine_pos_embedding(self, centroids: torch.Tensor, *args) -> torch.Tensor:
        """Compute fixed sine temporal embeddings per dimension (x,y)

        Args:
            centroids: the input centroids for either the x,y dimension represented
            by fraction of distance of original image that the instance centroid lies at;
             of shape (N,) or (N,1) where N = # of query tokens (i.e. instances)
             values between [0,1]

        Returns:
            an n_instances x D embedding representing the temporal embedding.
        """
        d = self.features
        n = self.temperature

        positions = centroids.unsqueeze(1)
        temp_lookup = torch.zeros(len(centroids), d, device=centroids.device)

        denominators = torch.pow(
            n, 2 * torch.arange(0, d // 2, device=centroids.device) / d
        )  # 10000^(2i/d_model), i is the index of embedding
        temp_lookup[:, 0::2] = torch.sin(
            positions / denominators
        )  # sin(pos/10000^(2i/d_model))
        temp_lookup[:, 1::2] = torch.cos(
            positions / denominators
        )  # cos(pos/10000^(2i/d_model))

        return temp_lookup  # .view(len(times), self.features)

    def _sine_box_embedding(self, boxes: torch.Tensor, *args) -> torch.Tensor:
        """Compute sine positional embeddings for boxes using given parameters.

         Args:
             boxes: the input boxes of shape N, n_anchors, 4 or B, N, n_anchors, 4
                    where the last dimension is the bbox coords in [y1, x1, y2, x2].
                    (Note currently `B=batch_size=1`).

        Returns:
             torch.Tensor, the sine positional embeddings
             (embedding[:, 4i] = sin(x)
              embedding[:, 4i+1] = cos(x)
              embedding[:, 4i+2] = sin(y)
              embedding[:, 4i+3] = cos(y)
              )
        """
        if self.scale is not None and self.normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if len(boxes.size()) == 3:
            boxes = boxes.unsqueeze(0)

        if self.normalize:
            boxes = boxes / (boxes[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.features // 4, dtype=torch.float32)

        dim_t = self.temperature ** (
            2 * self._torch_int_div(dim_t, 2) / (self.features // 4)
        )

        # (b, n_t, n_anchors, 4, D//4)
        pos_emb = boxes[:, :, :, :, None] / dim_t.to(boxes.device)

        pos_emb = torch.stack(
            (pos_emb[:, :, :, :, 0::2].sin(), pos_emb[:, :, :, :, 1::2].cos()), dim=4
        )
        pos_emb = pos_emb.flatten(2).squeeze(0)  # (N_t, n_anchors * D)

        pos_emb = self.mlp(pos_emb)

        pos_emb = pos_emb.view(boxes.shape[1], self.features)

        return pos_emb

    def _sine_temp_embedding(self, times: torch.Tensor, *args) -> torch.Tensor:
        """Compute fixed sine temporal embeddings.

        Args:
            times: the input times of shape (N,) or (N,1) where N = (sum(instances_per_frame))
            which is the frame index of the instance relative
            to the batch size
            (e.g. `torch.tensor([0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2,..., B, B, ...B])`).

        Returns:
            an n_instances x D embedding representing the temporal embedding.
        """
        T = times.int().max().item() + 1
        d = self.features
        n = self.temperature

        positions = torch.arange(0, T).unsqueeze(1)
        temp_lookup = torch.zeros(T, d, device=times.device)

        denominators = torch.pow(
            n, 2 * torch.arange(0, d // 2) / d
        )  # 10000^(2i/d_model), i is the index of embedding
        temp_lookup[:, 0::2] = torch.sin(
            positions / denominators
        )  # sin(pos/10000^(2i/d_model))
        temp_lookup[:, 1::2] = torch.cos(
            positions / denominators
        )  # cos(pos/10000^(2i/d_model))

        temp_emb = temp_lookup[times.int()]
        return temp_emb  # .view(len(times), self.features)

    def _learned_pos_embedding(self, boxes: torch.Tensor, *args) -> torch.Tensor:
        """Compute learned positional embeddings for boxes using given parameters.

        Args:
            boxes: the input boxes of shape N x 4 or B x N x 4
                   where the last dimension is the bbox coords in [y1, x1, y2, x2].
                   (Note currently `B=batch_size=1`).

        Returns:
            torch.Tensor, the learned positional embeddings.
        """
        pos_lookup = self.lookup

        N, n_anchors, _ = boxes.shape
        boxes = boxes.view(N, n_anchors, 4)

        if self.over_boxes:
            xywh = boxes
        else:
            xywh = torch.cat(
                [
                    (boxes[:, :, 2:] + boxes[:, :, :2]) / 2,
                    (boxes[:, :, 2:] - boxes[:, :, :2]),
                ],
                dim=1,
            )

        left_ind, right_ind, left_weight, right_weight = self._compute_weights(xywh)
        f = pos_lookup.weight.shape[1]  # self.features // 4

        try:
            pos_emb_table = pos_lookup.weight.view(
                self.emb_num, n_anchors, 4, f
            )  # T x 4 x (D * 4)
        except RuntimeError as e:
            logger.exception(
                f"Hint: `n_points` ({self.n_points}) may be set incorrectly!"
            )
            logger.exception(e)
            raise (e)

        left_emb = pos_emb_table.gather(
            0,
            left_ind[:, :, :, None].to(pos_emb_table.device).expand(N, n_anchors, 4, f),
        )  # N x 4 x d
        right_emb = pos_emb_table.gather(
            0,
            right_ind[:, :, :, None]
            .to(pos_emb_table.device)
            .expand(N, n_anchors, 4, f),
        )  # N x 4 x d
        pos_emb = left_weight[:, :, :, None] * right_emb.to(
            left_weight.device
        ) + right_weight[:, :, :, None] * left_emb.to(right_weight.device)

        pos_emb = pos_emb.flatten(1)
        pos_emb = self.mlp(pos_emb)

        return pos_emb.view(N, self.features)

    def _learned_temp_embedding(self, times: torch.Tensor, *args) -> torch.Tensor:
        """Compute learned temporal embeddings for times using given parameters.

        Args:
            times: the input times of shape (N,) or (N,1) where N = (sum(instances_per_frame))
            which is the frame index of the instance relative
            to the batch size
            (e.g. `torch.tensor([0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2,..., B, B, ...B])`).

        Returns:
            torch.Tensor, the learned temporal embeddings.
        """
        temp_lookup = self.lookup
        N = times.shape[0]
        times = times / times.max()

        left_ind, right_ind, left_weight, right_weight = self._compute_weights(times)

        left_emb = temp_lookup.weight[
            left_ind.to(temp_lookup.weight.device)
        ]  # T x D --> N x D
        right_emb = temp_lookup.weight[right_ind.to(temp_lookup.weight.device)]

        temp_emb = left_weight[:, None] * right_emb.to(
            left_weight.device
        ) + right_weight[:, None] * left_emb.to(right_weight.device)

        return temp_emb.view(N, self.features)

    def _compute_weights(self, data: torch.Tensor, *args) -> tuple[torch.Tensor, ...]:
        """Compute left and right learned embedding weights.

        Args:
            data: the input data (e.g boxes or times).

        Returns:
            A torch.Tensor for each of the left/right indices and weights, respectively
        """
        data = data * self.emb_num

        left_ind = data.clamp(min=0, max=self.emb_num - 1).long()  # N x 4
        right_ind = (left_ind + 1).clamp(min=0, max=self.emb_num - 1).long()  # N x 4

        left_weight = data - left_ind.float()  # N x 4

        right_weight = 1.0 - left_weight

        return left_ind, right_ind, left_weight, right_weight


def _pos_embed_fourier1d_init(cutoff, n):
    """Create a tensor of shape (1,n) of fourier frequency coefficients."""
    return torch.exp(torch.linspace(0, -math.log(cutoff), n)).unsqueeze(0)


class FourierPositionalEmbeddings(torch.nn.Module):
    """Fourier positional embeddings."""

    def __init__(
        self,
        n_components: int,
        d_model: int,
    ):
        """Initialize Fourier positional embeddings.

        Args:
            n_components: Number of frequencies for each dimension.
            d_model: Model dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.n_components = n_components
        self.freq = torch.nn.Parameter(
            _pos_embed_fourier1d_init(self.d_model, n_components)
        )

    def forward(self, seq_positions: torch.Tensor):
        """Compute learnable fourier coefficients for each spatial/temporal position.

        Args:
            seq_positions: tensor of shape (num_queries,)

        Returns:
            tensor of shape (num_queries, embed_dim)
        """
        freq = self.freq.to(seq_positions.device)
        # seq_positions is of shape (num_queries,) but needs to be (1,num_queries,1)
        embed = torch.cat(
            (
                torch.sin(
                    0.5 * math.pi * seq_positions.unsqueeze(-1).unsqueeze(0) * freq
                ),
                torch.cos(
                    0.5 * math.pi * seq_positions.unsqueeze(-1).unsqueeze(0) * freq
                ),
            ),
            axis=-1,
        ) / math.sqrt(
            len(freq)
        )  # (B,N,2*n_components)

        if self.d_model % self.n_components != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by number of Fourier components n_components ({self.n_components})"
            )

        # tile until shape is (B,N,embed_dim) to multiply with input queries/keys
        embed = embed.repeat(
            1, 1, self.d_model // (2 * self.n_components)
        )  # 2*n_components to account for sin/cos

        return embed
