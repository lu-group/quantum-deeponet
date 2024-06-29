import torch 
import numpy as np
import deepxde as dde
import sys
# from VanillaNet import OrthoONetCartesianProd
import os
os.environ['DDE_BACKEND'] = 'pytorch'
from classical_orthogonal_deeponet import OrthoONetCartesianProd
from physics_informed_zcs import PDE_OP, Model, LazyGrad
    
class  OrthoONetCartesianProd_DBC(OrthoONetCartesianProd):
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        regularization=None,
    ):
        super().__init__(layer_sizes_branch,
                        layer_sizes_trunk,
                        activation,
                        regularization)
        
    def dirichlet(self,x):
        # return torch.sin(torch.pi*x)
        return x
        
    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        if self._branch_transform is not None:
            x_func = self._branch_transform(x_func)
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        
        # print('x_loc',x_loc)
        multiplyer = self.dirichlet(x_loc) #*(50,1)

        if self._trunk_transform is not None:
            x_loc = self._trunk_transform(x_loc)
        x_loc = self.trunk(x_loc)
        x_loc = self.activation_trunk(x_loc)
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)  #*(100,10)X(50,10)->(100,50)
        # Add bias
        x += self.b
        x = x * multiplyer.reshape((1,-1))
        return x
    

def main():
    def equation_zcs(zcs_parameters,u,f):
        lazy_grad = LazyGrad(zcs_parameters, u)
        u_x = lazy_grad.compute((1,))
        return - u_x +  f

    # Domain is interval [0, 1]
    geom = dde.geometry.Interval(0, 1)
    
    # Define PDE
    pde = dde.data.PDE(geom, equation_zcs, [],num_domain=100, num_boundary=0)

    '''auxiliary_var_function: A function that inputs `train_x` or `test_x` and outputs auxiliary variables.'''

    # Function space for f(x) are polynomials
    space = dde.data.GRF(1, kernel = 'RBF', length_scale = 0.2, N = 1000, interp = 'cubic') #! in this example we use GRF

    # Choose evaluation points
    num_eval_points = 100
    evaluation_points = geom.uniform_points(num_eval_points, boundary=True)

    # Define PDE operator
    pde_op = PDE_OP(
        pde,
        space,
        evaluation_points,
        n_pca=19,
        num_function=10000,
        num_test=1000,
        batch_size=2000,
        pca_save_path='classical_training/pca_model.joblib'
    )

    # Setup DeepONet
    dim_x = 1
    n_pca=19
    p = 20
    net = OrthoONetCartesianProd_DBC(
        [n_pca+1,20,20,20,20,p],
        [dim_x+1,20,20,20,20,p],
        activation="tanh",
    )

    net.apply_branch_transform(pde_op.branch_transform, min=-40, max=40)
    net.apply_trunk_transform(pde_op.trunk_transform)

    model = Model(pde_op, net)
    model.compile('adam', lr=0.0001)
    losshistory, train_state = model.train(iterations=200000)
    dde.utils.external.save_loss_history(losshistory,r'classical_training/loss_history.txt' )
    model.save(r'classical_training/model_checkpoint')


    dde_model = model.net
    for name,param in dde_model.named_parameters():
        np.savetxt(fr'classical_training/{name}.txt',param.cpu().detach().numpy())

if __name__ == "__main__":
    main()