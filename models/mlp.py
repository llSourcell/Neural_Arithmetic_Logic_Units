import math
import torch.nn as nn

from .utils import str2act


class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP).

    Also known as a Fully-Connected Network (FCN). This
    implementation assumes that all hidden layers have
    the same hidden size and the same activation function.

    Attributes:
        num_layers: the number of layers in the network.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
        activation: the activation function.
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, activation='relu'):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = str2act(activation)

        nonlin = True
        if self.activation is None:
            nonlin = False

        layers = []
        for i in range(num_layers - 1):
            layers.extend(
                self._layer(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim,
                    nonlin,
                )
            )
        layers.extend(self._layer(hidden_dim, out_dim, False))

        self.model = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def _layer(self, in_dim, out_dim, activation=True):
        if activation:
            return [
                nn.Linear(in_dim, out_dim),
                self.activation,
            ]
        else:
            return [
                nn.Linear(in_dim, out_dim),
            ]

    def forward(self, x):
        out = self.model(x)
        return out
