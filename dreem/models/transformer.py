"""DETR Transformer class.

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

* Modified from https://github.com/facebookresearch/detr/blob/main/models/transformer.py
* Modified from https://github.com/xingyizhou/GTR/blob/master/gtr/modeling/roi_heads/transformer.py
* Modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
    * added fixed embeddings over boxes
"""

from dreem.io import AssociationMatrix
from dreem.models.attention_head import ATTWeightHead
from dreem.models import Embedding, FourierPositionalEmbeddings
from dreem.models.mlp import MLP
from dreem.models.model_utils import get_boxes, get_times
from torch import nn
import copy
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

# todo: add named tensors
# todo: add flash attention


class Transformer(torch.nn.Module):
    """Transformer class."""

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        norm: bool = False,
        num_layers_attn_head: int = 2,
        dropout_attn_head: float = 0.1,
        embedding_meta: dict | None = None,
        return_embedding: bool = False,
        decoder_self_attn: bool = False,
    ) -> None:
        """Initialize Transformer.

        Args:
            d_model: The number of features in the encoder/decoder inputs.
            nhead: The number of heads in the transfomer encoder/decoder.
            num_encoder_layers: The number of encoder-layers in the encoder.
            num_decoder_layers: The number of decoder-layers in the decoder.
            dropout: Dropout value applied to the output of transformer layers.
            activation: Activation function to use.
            return_intermediate_dec: Return intermediate layers from decoder.
            norm: If True, normalize output of encoder and decoder.
            num_layers_attn_head: The number of layers in the attention head.
            dropout_attn_head: Dropout value for the attention_head.
            embedding_meta: Metadata for positional embeddings. See below.
            return_embedding: Whether to return the positional embeddings
            decoder_self_attn: If True, use decoder self attention.

                More details on `embedding_meta`:
                    By default this will be an empty dict and indicate
                    that no positional embeddings should be used. To use the positional embeddings
                    pass in a dictionary containing a "pos" and "temp" key with subdictionaries for correct parameters ie:
                    {"pos": {'mode': 'learned', 'emb_num': 16, 'over_boxes: 'True'},
                    "temp": {'mode': 'learned', 'emb_num': 16}}. (see `dreem.models.embeddings.Embedding.EMB_TYPES`
                    and `dreem.models.embeddings.Embedding.EMB_MODES` for embedding parameters).
        """
        super().__init__()

        self.d_model = dim_feedforward = feature_dim_attn_head = d_model

        self.embedding_meta = embedding_meta
        self.return_embedding = return_embedding

        self.pos_emb = Embedding(emb_type="off", mode="off", features=self.d_model)
        self.temp_emb = Embedding(emb_type="off", mode="off", features=self.d_model)

        if self.embedding_meta:
            self.embedding_agg_method = (
                embedding_meta["embedding_agg_method"]
                if "embedding_agg_method" in embedding_meta
                else "average"
            )
            self.use_fourier = (
                embedding_meta["use_fourier"]
                if "use_fourier" in embedding_meta
                else False
            )
            self.fourier_n_components = (
                embedding_meta["fourier_n_components"]
                if "fourier_n_components" in embedding_meta
                else self.d_model // 2
            )
            if "pos" in self.embedding_meta:
                pos_emb_cfg = self.embedding_meta["pos"]
                if pos_emb_cfg:
                    self.pos_emb = Embedding(
                        emb_type="pos",
                        features=self.d_model,
                        embedding_agg_method=self.embedding_agg_method,
                        use_fourier=self.use_fourier,
                        fourier_n_components=self.fourier_n_components,
                        **pos_emb_cfg,
                    )  # agg method must be the same for pos and temp embeddings
            if "temp" in self.embedding_meta:
                temp_emb_cfg = self.embedding_meta["temp"]
                if temp_emb_cfg:
                    self.temp_emb = Embedding(
                        emb_type="temp",
                        features=self.d_model,
                        embedding_agg_method=self.embedding_agg_method,
                        use_fourier=self.use_fourier,
                        fourier_n_components=self.fourier_n_components,
                        **temp_emb_cfg,
                    )
        else:
            self.embedding_meta = {}
            self.embedding_agg_method = None
            self.use_fourier = False
            self.fourier_n_components = None

        if self.use_fourier:
            self.fourier_emb = FourierPositionalEmbeddings(
                self.fourier_n_components, self.d_model
            )

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            norm,
            **self.embedding_meta,
        )

        encoder_norm = nn.LayerNorm(d_model) if (norm) else None

        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm, **self.embedding_meta
        )

        # Transformer Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            norm,
            decoder_self_attn,
            **self.embedding_meta,
        )

        decoder_norm = nn.LayerNorm(d_model) if (norm) else None

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
            decoder_norm,
            **self.embedding_meta,
        )

        # Transformer attention head
        self.attn_head = ATTWeightHead(
            feature_dim=feature_dim_attn_head,
            num_layers=num_layers_attn_head,
            dropout=dropout_attn_head,
            **self.embedding_meta,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model weights from xavier distribution."""
        for p in self.parameters():
            if not torch.nn.parameter.is_lazy(p) and p.dim() > 1:
                try:
                    nn.init.xavier_uniform_(p)
                except ValueError as e:
                    print(f"Failed Trying to initialize {p}")
                    raise (e)

    def forward(
        self,
        ref_instances: list["dreem.io.Instance"],
        query_instances: list["dreem.io.Instance"] | None = None,
    ) -> list[AssociationMatrix]:
        """Execute a forward pass through the transformer and attention head.

        Args:
            ref_instances: A list of instance objects (See `dreem.io.Instance` for more info.)
            query_instances: An set of instances to be used as decoder queries.

        Returns:
            asso_output: A list of torch.Tensors of shape (L, n_query, total_instances) where:
                L: number of decoder blocks
                n_query: number of instances in current query/frame
                total_instances: number of instances in window
        """
        ref_features = torch.cat(
            [instance.features for instance in ref_instances], dim=0
        ).unsqueeze(0)

        # instances_per_frame = [frame.num_detected for frame in frames]
        total_instances = len(ref_instances)
        embed_dim = ref_features.shape[-1]
        # print(f'T: {window_length}; N: {total_instances}; N_t: {instances_per_frame} n_reid: {reid_features.shape}')
        ref_boxes = get_boxes(ref_instances)  # total_instances, 4
        ref_boxes = torch.nan_to_num(ref_boxes, -1.0)
        ref_times, query_times = get_times(ref_instances, query_instances)

        batch_size, total_instances, embed_dim = ref_features.shape
        ref_features = ref_features.permute(
            1, 0, 2
        )  # (total_instances, batch_size, embed_dim)
        encoder_queries = ref_features

        # apply fourier embeddings
        if "use_fourier" in self.embedding_meta and self.embedding_meta["use_fourier"]:
            encoder_queries = apply_fourier_embeddings(
                encoder_queries,
                ref_boxes,
                ref_times,
                self.fourier_n_components,
                self.fourier_emb,
            )

        # (encoder_features, ref_pos_emb, ref_temp_emb) \
        encoder_features, pos_emb_traceback, temp_emb_traceback = self.encoder(
            encoder_queries,
            embedding_map={"pos": self.pos_emb, "temp": self.temp_emb},
            boxes=ref_boxes,
            times=ref_times,
            embedding_agg_method=self.embedding_agg_method,
        )  # (total_instances, batch_size, embed_dim) or
        # (3*total_instances,batch_size,embed_dim) if using stacked embeddings

        if self.return_embedding:
            for i, instance in enumerate(ref_instances):
                if self.embedding_agg_method == "average":
                    ref_pos_emb = pos_emb_traceback[0][i]  # array
                else:
                    ref_pos_emb = {
                        "x": pos_emb_traceback[0][0][i],
                        "y": pos_emb_traceback[1][0][i],
                    }  # dict

                instance.add_embedding("pos", ref_pos_emb)  # can be an array or a dict
                instance.add_embedding("temp", temp_emb_traceback)

        # -------------- Begin decoder --------------- #
        # prepare queries, embedding position ids, etc

        # for inference, query_instances is not None
        if query_instances is not None:
            n_query = len(query_instances)
            query_features = torch.cat(
                [instance.features for instance in query_instances], dim=0
            ).unsqueeze(0)

            query_features = query_features.permute(
                1, 0, 2
            )  # (n_query, batch_size, embed_dim)

            # just get boxes, we already have query_times from above
            query_boxes = get_boxes(query_instances)
            query_boxes = torch.nan_to_num(query_boxes, -1.0)
        else:  # for training, query_instances is None so just pass in the ref data
            n_query = total_instances
            query_instances = ref_instances
            query_features = ref_features
            query_boxes = ref_boxes
            query_times = ref_times

        decoder_features, pos_emb_traceback, temp_emb_traceback = self.decoder(
            query_features,
            encoder_features,
            embedding_map={"pos": self.pos_emb, "temp": self.temp_emb},
            boxes=query_boxes,
            times=query_times,
            enc_boxes=ref_boxes,
            enc_times=ref_times,
            embedding_agg_method=self.embedding_agg_method,
        )  # (L, n_query, batch_size, embed_dim)

        if self.return_embedding:
            for i, instance in enumerate(ref_instances):
                if self.embedding_agg_method == "average":
                    ref_pos_emb = pos_emb_traceback[0][i]  # array
                else:
                    ref_pos_emb = {
                        "x": pos_emb_traceback[0][0][i],
                        "y": pos_emb_traceback[1][0][i],
                    }  # dict

                instance.add_embedding("pos", ref_pos_emb)  # can be an array or a dict
                instance.add_embedding("temp", temp_emb_traceback)

        decoder_features = decoder_features.transpose(
            1, 2
        )  # # (L, batch_size, n_query, embed_dim) or ((L, batch_size, 3*n_query, embed_dim)) if using stacked embeddings
        encoder_features = encoder_features.permute(1, 0, 2)
        # (batch_size, total_instances, embed_dim) or (batch_size, 3*total_instances, embed_dim)

        asso_output = []
        for frame_features in decoder_features:
            # attn_head handles the 3x queries that can come out of the encoder/decoder if using stacked embeddings
            # n_query should be the number of instances in the last frame if running inference,
            # or number of ref instances for training. total_instances is always the number of reference instances
            asso_matrix = self.attn_head(frame_features, encoder_features).view(
                n_query, total_instances
            )  # call to view() just removes the batch dimension; output of attn_head is (1,n_query,total_instances)
            asso_matrix = AssociationMatrix(asso_matrix, ref_instances, query_instances)

            asso_output.append(asso_matrix)

        # (L=1, n_query, total_instances)
        return asso_output


class TransformerEncoderLayer(nn.Module):
    """A single transformer encoder layer."""

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        norm: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a transformer encoder layer.

        Args:
            d_model: The number of features in the encoder inputs.
            nhead: The number of heads for the encoder.
            dim_feedforward: Dimension of the feedforward layers of encoder.
            dropout: Dropout value applied to the output of encoder.
            activation: Activation function to use.
            norm: If True, normalize output of encoder.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        queries: torch.Tensor,
        orig_queries: torch.Tensor,
        embedding_map: Dict[str, Embedding],
    ) -> torch.Tensor:
        """Execute a forward pass of the encoder layer.

        Args:
            queries: Input sequence for encoder (n_query, batch_size, embed_dim);
                    data is already transformed with embedding
            orig_queries: Original query data before embedding (n_query, batch_size, embed_dim).
                        Used for rope embedding method since rope only applied to q,k not v

        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
        """
        # rope only applied to q,k not v
        if embedding_map["pos"].mode == "rope" and embedding_map["temp"].mode == "rope":
            v = orig_queries
        else:
            v = queries

        attn_features = self.self_attn(
            query=queries,
            key=queries,
            value=v,
        )[0]

        orig_queries = orig_queries + self.dropout1(attn_features)
        orig_queries = self.norm1(orig_queries)
        projection = self.linear2(
            self.dropout(self.activation(self.linear1(orig_queries)))
        )
        orig_queries = orig_queries + self.dropout2(projection)
        orig_queries = self.norm2(orig_queries)

        return orig_queries


class TransformerDecoderLayer(nn.Module):
    """A single transformer decoder layer."""

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        norm: bool = False,
        decoder_self_attn: bool = False,
        **kwargs,
    ) -> None:
        """Initialize transformer decoder layer.

        Args:
            d_model: The number of features in the decoder inputs.
            nhead: The number of heads for the decoder.
            dim_feedforward: Dimension of the feedforward layers of decoder.
            dropout: Dropout value applied to the output of decoder.
            activation: Activation function to use.
            norm: If True, normalize output of decoder.
            decoder_self_attn: If True, use decoder self attention
        """
        super().__init__()

        self.decoder_self_attn = decoder_self_attn

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if self.decoder_self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        decoder_queries: torch.Tensor,
        encoder_features: torch.Tensor,
        orig_decoder_queries: torch.Tensor,
        embedding_map: Dict[str, Embedding],
        boxes: torch.Tensor,
        times: torch.Tensor,
        enc_boxes: torch.Tensor,
        enc_times: torch.Tensor,
        embedding_agg_method: str = None,
    ) -> torch.Tensor:
        """Execute forward pass of decoder layer.

        Args:
            decoder_queries: Target sequence for decoder to generate (n_query, batch_size, embed_dim);
                              data is already transformed with embedding
            encoder_features: Output from encoder, that decoder uses to attend to relevant
                parts of input sequence (total_instances, batch_size, embed_dim)
            orig_decoder_queries: Original query data before embedding (n_query, batch_size, embed_dim).
                        Used for rope embedding method since rope only applied to q,k not v
            embedding_map: Dict of Embedding objects defining the pos/temp embeddings to be applied to
                        the input data before it passes to the DecoderLayer
            boxes: Bounding box based embedding ids of shape (n_query, batch_size, 4)
            times: Times based embedding ids of shape (n_query,)
            enc_boxes: Bounding box based embedding ids of shape (total_instances, batch_size, 4) for encoder features
            enc_times: Times based embedding ids of shape (total_instances,) for encoder features
            embedding_agg_method: method of aggregation of embeddings e.g. stack/concatenate/average

        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
        """
        # rope only applied to q,k not v
        if embedding_map["pos"].mode == "rope" and embedding_map["temp"].mode == "rope":
            v = orig_decoder_queries
        else:
            v = decoder_queries

        if self.decoder_self_attn:
            # self attn; decoder_queries has embeddings applied, orig_decoder_queries does not
            # for rope, value should not be embedded so use value=orig_decoder_queries
            self_attn_features = self.self_attn(
                query=decoder_queries, key=decoder_queries, value=v
            )[0]
            orig_decoder_queries = orig_decoder_queries + self.dropout1(
                self_attn_features
            )
            orig_decoder_queries = self.norm1(orig_decoder_queries)

        # embeddings need to be reapplied to decoder queries between self attention and cross attention;
        # Reference: https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L187-L233

        q_x_attn, *_ = apply_embeddings(
            orig_decoder_queries, embedding_map, boxes, times, embedding_agg_method
        )  # queries from decoder
        # remember, encoder features should have embeddings applied with encoder position ids from enc_boxes, enc_times
        k_x_attn, *_ = apply_embeddings(
            encoder_features, embedding_map, enc_boxes, enc_times, embedding_agg_method
        )  # keys from encoder

        # cross attention
        # for rope, value should not be embedded so use value=encoder_features
        if embedding_map["pos"].mode == "rope" and embedding_map["temp"].mode == "rope":
            v = encoder_features
        else:
            v = k_x_attn

        x_attn_features = self.multihead_attn(
            query=q_x_attn,  # (n_query, batch_size, embed_dim)
            key=k_x_attn,  # (total_instances, batch_size, embed_dim)
            value=v,  # (total_instances, batch_size, embed_dim)
        )[
            0
        ]  # (n_query, batch_size, embed_dim)

        orig_decoder_queries = orig_decoder_queries + self.dropout2(
            x_attn_features
        )  # (n_query, batch_size, embed_dim)
        orig_decoder_queries = self.norm2(
            orig_decoder_queries
        )  # (n_query, batch_size, embed_dim)
        projection = self.linear2(
            self.dropout(self.activation(self.linear1(orig_decoder_queries)))
        )  # (n_query, batch_size, embed_dim)
        orig_decoder_queries = orig_decoder_queries + self.dropout3(
            projection
        )  # (n_query, batch_size, embed_dim)
        orig_decoder_queries = self.norm3(orig_decoder_queries)

        return orig_decoder_queries  # (n_query, batch_size, embed_dim)


class TransformerEncoder(nn.Module):
    """A transformer encoder block composed of encoder layers."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Initialize transformer encoder.

        Args:
            encoder_layer: An instance of the TransformerEncoderLayer.
            num_layers: The number of encoder layers to be stacked.
            norm: The normalization layer to be applied.
        """
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm if norm is not None else nn.Identity()

    def forward(
        self,
        queries: torch.Tensor,
        embedding_map: Dict[str, Embedding],
        boxes: torch.Tensor,
        times: torch.Tensor,
        embedding_agg_method: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute a forward pass of encoder layer. Computes and applies embeddings before input to EncoderLayer

        Args:
            queries: The input tensor of shape (n_query, batch_size, embed_dim).
            embedding_map: Dict of Embedding objects defining the pos/temp embeddings to be applied to
                        the input data before it passes to the EncoderLayer
            boxes: Bounding box based embedding ids of shape (n_query, batch_size, 4)
            times:
            embedding_agg_method:

        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
        """

        for layer in self.layers:
            # compute embeddings and apply to the input queries
            queries, orig_queries, pos_emb_traceback, temp_emb_traceback = (
                apply_embeddings(
                    queries, embedding_map, boxes, times, embedding_agg_method
                )
            )
            # pass through EncoderLayer
            encoder_features = layer(queries, orig_queries, embedding_map)

        encoder_features = self.norm(encoder_features)

        return encoder_features, pos_emb_traceback, temp_emb_traceback


class TransformerDecoder(nn.Module):
    """Transformer Decoder Block composed of Transformer Decoder Layers."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        return_intermediate: bool = False,
        norm: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Initialize transformer decoder block.

        Args:
            decoder_layer: An instance of TransformerDecoderLayer.
            num_layers: The number of decoder layers to be stacked.
            return_intermediate: Return intermediate layers from decoder.
            norm: The normalization layer to be applied.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm if norm is not None else nn.Identity()

    def forward(
        self,
        decoder_queries: torch.Tensor,
        encoder_features: torch.Tensor,
        embedding_map: Dict[str, Embedding],
        boxes: torch.Tensor,
        times: torch.Tensor,
        enc_boxes: torch.Tensor,
        enc_times: torch.Tensor,
        embedding_agg_method: str = None,
    ) -> torch.Tensor:
        """Execute a forward pass of the decoder block.

        Args:
            decoder_queries: Query sequence for decoder to generate (n_query, batch_size, embed_dim).
            encoder_features: Output from encoder, that decoder uses to attend to relevant
                parts of input sequence (total_instances, batch_size, embed_dim)


        Returns:
            The output tensor of shape (L, n_query, batch_size, embed_dim).
        """
        decoder_features = decoder_queries
        intermediate = []

        # since the encoder output doesn't change for any number of decoder layer inputs,
        # we can process its embedding outside the loop
        # if embedding_agg_method == "average":
        #     encoder_features, *_ = apply_embeddings(
        #         encoder_features,embedding_map,enc_boxes,enc_times,embedding_agg_method,
        #     )
        # ^ redundant since encoder features get embeddings added again in each decoder layer

        for layer in self.layers:
            (
                decoder_features,
                orig_decoder_queries,
                pos_emb_traceback,
                temp_emb_traceback,
            ) = apply_embeddings(
                decoder_features, embedding_map, boxes, times, embedding_agg_method
            )
            decoder_features = layer(
                decoder_features,
                encoder_features,
                orig_decoder_queries,
                embedding_map,
                boxes,
                times,
                enc_boxes,
                enc_times,
                embedding_agg_method,
            )

            if self.return_intermediate:
                intermediate.append(self.norm(decoder_features))

        decoder_features = self.norm(decoder_features)

        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(decoder_features)
            return torch.stack(intermediate), pos_emb_traceback, temp_emb_traceback

        return decoder_features.unsqueeze(0), pos_emb_traceback, temp_emb_traceback


def apply_fourier_embeddings(
    queries: torch.Tensor,
    boxes: torch.Tensor,
    times: torch.Tensor,
    fourier_n_components: int,
    fourier_emb: FourierPositionalEmbeddings,
) -> torch.Tensor:
    """Applies Fourier positional embeddings to input queries
    Args:
        queries: Input queries of shape (n_query, batch_size, embed_dim)
        boxes: Bounding box based embedding ids of shape (n_query, n_anchors, 4)
        times: Times based embedding ids of shape (n_query,)
        fourier_n_components: number of fourier components to use for fourier positional embeddings
        fourier_emb: FourierPositionalEmbeddings object
    Returns:
        Tensor: Input queries with Fourier positional embeddings added - shape (n_query, batch_size, embed_dim)
    """

    # queries is of shape (n_query, batch_size, embed_dim); transpose for embeddings
    queries = queries.permute(
        1, 0, 2
    )  # queries is now of shape (batch_size, n_query, embed_dim)

    ref_x, ref_y = spatial_emb_from_bb(boxes)
    t_emb = fourier_emb(times)
    x_emb = fourier_emb(ref_x)
    y_emb = fourier_emb(ref_y)

    queries_cat = torch.cat((t_emb, x_emb, y_emb, queries), dim=-1)

    # queries is shape (batch_size, n_query, embed_dim + 3*embed_dim)
    proj = nn.Linear(queries_cat.shape[-1], queries.shape[-1]).to(queries_cat.device)
    norm = nn.LayerNorm(queries.shape[-1]).to(queries_cat.device)

    queries = proj(queries_cat)
    queries = norm(queries)

    return queries.permute(1, 0, 2)


def apply_embeddings(
    queries: torch.Tensor,
    embedding_map: Dict[str, Embedding],
    boxes: torch.Tensor,
    times: torch.Tensor,
    embedding_agg_method: str,
):
    """Applies embeddings to input queries for various aggregation methods. This function
    is called from the transformer encoder and decoder

    Args:
        queries: The input tensor of shape (n_query, batch_size, embed_dim).
        embedding_map: Dict of Embedding objects defining the pos/temp embeddings to be applied
        to the input data
        boxes: Bounding box based embedding ids of shape (n_query, n_anchors, 4)
        times: Times based embedding ids of shape (n_query,)
        embedding_agg_method: method of aggregation of embeddings e.g. stack/concatenate/average
    """

    pos_emb, temp_emb = embedding_map["pos"], embedding_map["temp"]
    orig_queries = queries.clone().detach()
    # queries is of shape (n_query, batch_size, embed_dim); transpose for embeddings
    queries = queries.permute(
        1, 0, 2
    )  # queries is shape (batch_size, n_query, embed_dim)
    # calculate temporal embeddings and transform queries
    queries_t, ref_temp_emb = temp_emb(queries, times)

    if embedding_agg_method is None:
        pos_emb_traceback = (torch.zeros_like(queries),)
        queries_avg = queries_t = queries_x = queries_y = None
    else:
        # if avg. of temp and pos, need bounding boxes; bb only used for method "average"
        if embedding_agg_method == "average":
            _, ref_pos_emb = pos_emb(queries, boxes)
            ref_emb = (ref_pos_emb + ref_temp_emb) / 2
            queries_avg = queries + ref_emb
            queries_t = queries_x = queries_y = None
            pos_emb_traceback = (ref_pos_emb,)
        else:
            # calculate embedding array for x,y from bb centroids; ref_x, ref_y of shape (n_query,)
            ref_x, ref_y = spatial_emb_from_bb(boxes)
            # forward pass of Embedding object transforms input queries with embeddings
            queries_x, ref_pos_emb_x = pos_emb(queries, ref_x)
            queries_y, ref_pos_emb_y = pos_emb(queries, ref_y)
            queries_avg = None  # pass dummy var in to collate_queries
            pos_emb_traceback = (ref_pos_emb_x, ref_pos_emb_y)

    # concatenate or stack the queries (avg. method done above since it applies differently)
    queries = collate_queries(
        (queries_avg, queries_t, queries_x, queries_y, queries), embedding_agg_method
    )
    # transpose for input to EncoderLayer to (n_queries, batch_size, embed_dim)
    queries = queries.permute(1, 0, 2)

    return queries, orig_queries, pos_emb_traceback, ref_temp_emb


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Generate repeated clones of same layer type.

    Args:
        module: The module to be copied.
        N: The number of copies to be made.

    Returns:
        A ModuleList containing N copies of the given module.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> callable:
    """Choose activation function to be used.

    Args:
        activation: The string name of the activation function to use

    Returns:
        The appropriate activation function
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def collate_queries(
    queries: Tuple[torch.Tensor], embedding_agg_method: str
) -> torch.Tensor:
    """Aggregates queries transformed by embeddings

    Args:
        _queries: 5-tuple of queries (already transformed by embeddings) for _, x, y, t, original input
                  each of shape (batch_size, n_query, embed_dim)
        embedding_agg_method: String representing the aggregation method for embeddings

    Returns:
        Tensor of aggregated queries of shape; can be concatenated (increased length of tokens),
            stacked (increased number of tokens), or averaged (original token number and length)
    """

    queries_avg, queries_t, queries_x, queries_y, orig_queries = queries

    if embedding_agg_method == "average":
        collated_queries = queries_avg
    elif embedding_agg_method == "stack":
        # (t1,t2,t3...),(x1,x2,x3...),(y1,y2,y3...)
        # stacked is of shape (batch_size, 3*n_query, embed_dim)
        collated_queries = torch.cat((queries_t, queries_x, queries_y), dim=1)
    elif embedding_agg_method == "concatenate":
        mlp = MLP(
            input_dim=queries_t.shape[-1] * 3,  # t,x,y
            hidden_dim=queries_t.shape[-1] * 6,  # not applied when num_layers=1
            output_dim=queries_t.shape[-1],
            num_layers=1,
            dropout=0.0,
        )
        # collated_queries is of shape (batch_size, n_query, 3*embed_dim)
        collated_queries = torch.cat((queries_t, queries_x, queries_y), dim=2)
        # pass through MLP to project back into space of (batch_size, n_query, embed_dim)
        collated_queries = mlp(collated_queries)
    else:
        collated_queries = orig_queries

    return collated_queries


def spatial_emb_from_bb(bb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes embedding arrays for x,y spatial dimensions using centroids from bounding boxes

    Args:
        bb: Bounding boxes of shape (n_query, n_anchors, 4) from which to compute x,y centroids;
        each bounding box is [ymin, xmin, ymax, xmax]

    Returns:
        A tuple of tensors containing the emebdding array for x,y dimensions, each of shape (n_query,)
    """
    # compute avg of xmin,xmax and ymin,ymax
    return (
        bb[:, :, [1, 3]].mean(axis=2).squeeze(),
        bb[:, :, [0, 2]].mean(axis=2).squeeze(),
    )
