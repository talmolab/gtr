"""Multi-Layer Perceptron (MLP) module."""

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        """Initialize MLP.

        Args:
            input_dim: Dimensionality of the input features.
            hidden_dim: Number of units in the hidden layers.
            output_dim: Dimensionality of the output features.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        if self.num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(n, k)
                    # list concatenations to ensure layer shape compability
                    for n, k in zip([input_dim] + h, h + [output_dim])
                ]
            )
            if self.dropout > 0.0:
                self.dropouts = torch.nn.ModuleList(
                    [torch.nn.Dropout(dropout) for _ in range(self.num_layers - 1)]
                )
        else:
            self.layers = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor of shape (batch_size, num_instances, input_dim).

        Returns:
            Output tensor of shape (batch_size, num_instances, output_dim).
        """
        for i, layer in enumerate(self.layers):
            layer.to(x.device)
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1 and self.dropout > 0.0:
                x = self.dropouts[i](x)

        return x
