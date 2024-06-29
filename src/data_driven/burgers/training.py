import os
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
import numpy as np
from deepxde.backend import torch
import deepxde.nn.activations as activations
import deepxde.nn.initializers as initializers
from classical_orthogonal_NN import OrthoNN

# define ResNet architecture
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
    
def main():
    d1 = np.load(r'Burgers_train.npz',allow_pickle=True)
    x_train,y_train = (d1['X_train0'].astype(np.float32),d1['X_train1'].astype(np.float32)),d1['y_train'].astype(np.float32)
    d2 = np.load(r'Burgers_test.npz',allow_pickle=True)
    x_test,y_test = (d2['X_test0'].astype(np.float32),d2['X_test1'].astype(np.float32)),d2['y_test'].astype(np.float32)

    def periodic(x):
        return np.concatenate([np.cos(x[:,0:1]*2*np.pi),
                        np.sin(x[:,0:1]*2*np.pi),
                        np.cos(x[:,0:1]*4*np.pi),
                        np.sin(x[:,0:1]*4*np.pi),x[:,1:2]],axis=1)

    trunk_min = np.min( np.stack((np.min(periodic(x_train[1]), axis=0),np.min(periodic(x_test[1]), axis=0)),axis=0),axis=0)
    trunk_max = np.max( np.stack((np.max(periodic(x_train[1]), axis=0),np.max(periodic(x_test[1]), axis=0)),axis=0),axis=0)
    branch_min = np.min(np.stack((np.min(x_train[0], axis=0),np.min(x_test[0], axis=0)),axis=0),axis=0)
    branch_max = np.max( np.stack((np.max(x_train[0], axis=0),np.max(x_test[0], axis=0)),axis=0),axis=0)

    np.savez(r'input_transform.npz', trunk_min = trunk_min, trunk_max = trunk_max, branch_min = branch_min, branch_max = branch_max)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    def trunk_transform(x):
        d = x.shape[1]
        x = 2 * (x - trunk_min) / (trunk_max - trunk_min) - 1  # Rescale to [-1, 1]
        x = x / np.sqrt(d)
        x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
        return np.concatenate((x, x_d1), axis=1)

    def branch_transform(x):
        # For 1-dimensional input
        d = x.shape[1]
        x = 2 * (x - branch_min) / (branch_max - branch_min) - 1  # Rescale to [-1, 1]
        x = x / np.sqrt(d)
        x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
        return np.concatenate((x, x_d1), axis=1)

    x_train = (branch_transform(x_train[0]),trunk_transform(periodic(x_train[1])))
    x_test = (branch_transform(x_test[0]),trunk_transform(periodic(x_test[1])))
    data = dde.data.TripleCartesianProd(
        X_train = x_train, y_train = y_train, X_test = x_test, y_test = y_test
    )
    #choose network
    m = 20
    dim_x = 2
    net = OrthoONetCartesianProd(
        [m+1,20,20,20,20,20,20],
        [6,20,20,20,20,20,20],
        {'branch':'silu','trunk':'silu'}
    )

    model = dde.Model(data,net)
    model.compile('adam',lr=0.0005,metrics=['mean l2 relative error'])
    losshistory, train_state = model.train(iterations=30000,disregard_previous_best = True)

    dde.utils.external.save_loss_history(losshistory,r'classical_training/loss_history.txt' )
    dde_model = model.net
    model.save(r'classical_training/model_checkpoint')
    for name,param in dde_model.named_parameters():
        np.savetxt(fr'classical_training/{name}.txt',param.cpu().detach().numpy())

if __name__ == "__main__":
    main()