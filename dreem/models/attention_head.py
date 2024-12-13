"""Module containing different components of multi-head attention heads."""

import torch
from dreem.models.mlp import MLP

# todo: add named tensors


class ATTWeightHead(torch.nn.Module):
    """Single attention head."""

    def __init__(self, feature_dim: int, num_layers: int, dropout: float, **kwargs):
        """Initialize an instance of ATTWeightHead.

        Args:
            feature_dim: The dimensionality of input features.
            num_layers: The number of hidden layers in the MLP.
            dropout: Dropout probability.
            embedding_agg_method: how the embeddings are aggregated; average/stack/concatenate
        """
        super().__init__()
        if "embedding_agg_method" in kwargs:
            self.embedding_agg_method = kwargs["embedding_agg_method"]
        else:
            self.embedding_agg_method = None

        # if using stacked embeddings, use 1x1 conv with x,y,t embeddings as channels
        # ensures output represents ref instances by query instances
        if self.embedding_agg_method == "stack":
            self.q_proj = torch.nn.Conv1d(
                in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0
            )
            self.k_proj = torch.nn.Conv1d(
                in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0
            )
            self.attn_x = torch.nn.MultiheadAttention(feature_dim, 1)
            self.attn_y = torch.nn.MultiheadAttention(feature_dim, 1)
            self.attn_t = torch.nn.MultiheadAttention(feature_dim, 1)
        else:
            self.q_proj = MLP(
                feature_dim, feature_dim, feature_dim, num_layers, dropout
            )
            self.k_proj = MLP(
                feature_dim, feature_dim, feature_dim, num_layers, dropout
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the attention weights of a query tensor using the key tensor.

        Args:
            query: Input tensor of shape (batch_size, num_frame_instances, feature_dim).
            key: Input tensor of shape (batch_size, num_window_instances, feature_dim).

        Returns:
            Output tensor of shape (batch_size, num_frame_instances, num_window_instances).
        """
        batch_size, num_query_instances, feature_dim = query.size()
        num_window_instances = key.shape[1]

        # if stacked embeddings, create channels for each x,y,t embedding dimension
        # maps shape (1,num_instances*3,feature_dim) -> (num_instances,3,feature_dim)
        if self.embedding_agg_method == "stack":
            key_stacked = (
                key
                .view(batch_size, 3, num_window_instances // 3, feature_dim)
                .permute(0, 2, 1, 3)
                .squeeze(0) # keep as (num_instances*3, feature_dim)
            )
            key_orig = key.squeeze(0) # keep as (num_instances*3, feature_dim)

            query = (
                query.view(batch_size, 3, num_query_instances // 3, feature_dim)
                .permute(0, 2, 1, 3)
                .squeeze(0)
            )
            # pass t,x,y frame features through cross attention with entire encoder 3*num_window_instances tokens before MLP;
            # note order is t,x,y
            out_t, _ = self.attn_t(query=query[:,0,:], key=key_orig, value=key_orig)
            out_x, _ = self.attn_x(query=query[:,1,:], key=key_orig, value=key_orig)
            out_y, _ = self.attn_y(query=query[:,2,:], key=key_orig, value=key_orig)
            # combine each attention output to (num_instances, 3, feature_dim)
            collated = torch.stack((out_t, out_x, out_y), dim=0).permute(1,0,2)
            # mlp_out has shape (1, num_window_instances, feature_dim)
            mlp_out = self.q_proj(collated).transpose(1,0)

            # key, query of shape (num_instances, 3, feature_dim)
            # TODO: uncomment this if not using modified attention heads for t,x,y
            k = self.k_proj(key_stacked).transpose(1, 0)
            # q = self.q_proj(query).transpose(1, 0)
            # k,q of shape (num_instances, feature_dim)
            attn_weights = torch.bmm(mlp_out, k.transpose(1, 2))
        else:
            k = self.k_proj(key)
            q = self.q_proj(query)
            attn_weights = torch.bmm(q, k.transpose(1, 2))


        return attn_weights  # (B, N_t, N)
