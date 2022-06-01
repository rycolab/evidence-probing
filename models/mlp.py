from overrides import overrides
from torch import nn

from probekit.models.discriminative.neural_probe import NeuralProbeModel


class MLPS(nn.Sequential, NeuralProbeModel):
    """Compact MLP that supports backpacking."""

    def __init__(self, input_size, hidden_sizes, output_size, activation="tanh", flatten=False):
        super(MLPS, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = 1

        # Set activation function
        if activation == "relu":
            act = nn.ReLU
        elif activation == "tanh":
            act = nn.Tanh
        elif activation == "sigmoid":
            act = nn.Sigmoid
        else:
            raise ValueError('invalid activation')

        if flatten:
            self.add_module('flatten', nn.Flatten())

        if len(hidden_sizes) == 0:
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, self.output_size))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], self.output_size))

    @overrides
    def get_input_dim(self) -> int:
        return self.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self.output_size
