import torch
import deepxde as dde
import deepxde.nn.activations as activations
from classical_orthogonal_layer import OrthoLayer

# rectangular or square
class OrthoNN(dde.nn.pytorch.NN): 
    def __init__(self, layer_sizes, activation):  #e.g. [3,4,5,1] need data pre-processing
        super().__init__()
        # determine the activation function
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        self.layer_sizes = layer_sizes
        self.hidden_layers= torch.nn.ModuleList()
        for i in range(len(layer_sizes)-2):
            self.hidden_layers.append(OrthoLayer(layer_sizes[i],layer_sizes[i+1]))
        self.output_layer = torch.nn.Linear(layer_sizes[-2],layer_sizes[-1])        
    
    def forward(self,inputs):
        x = inputs

        # apply input transform when input is 1-dimention
        if self._input_transform is not None:
            x = self._input_transform(x)
    
        for i in range(len(self.layer_sizes)-2):
            norm = x.norm(dim=1, keepdim=True)
            x = x/norm

            x = self.hidden_layers[i](x)
            x = (
                self.activation[i](x)
                if isinstance(self.activation, list)
                else self.activation(x)
            )
        x = self.output_layer(x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x 