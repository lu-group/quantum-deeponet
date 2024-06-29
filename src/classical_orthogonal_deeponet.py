import torch
import deepxde as dde
import deepxde.nn.activations as activations
from classical_orthogonal_NN import OrthoNN

class OrthoONetCartesianProd(dde.nn.pytorch.NN):
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)

        self.branch = OrthoNN(layer_sizes_branch, activation_branch)
        self.trunk = OrthoNN(layer_sizes_trunk, self.activation_trunk)
        self.b = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        self.regularizer = regularization
        self._branch_transform = None
        self._trunk_transform = None

    #input tranformation
    def apply_branch_transform(self, transform):
        self._branch_transform = transform
    def apply_trunk_transform(self, transform):
        self._trunk_transform = transform

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        if self._branch_transform is not None:
            x_func = self._branch_transform(x_func)
        # Branch net to encode the input function
        x_func = self.branch(x_func)

        if self._trunk_transform is not None:
            x_loc = self._trunk_transform(x_loc)
        x_loc = self.trunk(x_loc)
        x_loc = self.activation_trunk(x_loc)
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)
        # Add bias
        x += self.b
        return x