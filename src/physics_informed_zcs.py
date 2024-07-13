import joblib
from typing import Tuple

import torch
import numpy as np
import deepxde as dde
import deepxde.backend as bkd
from sklearn.decomposition import PCA


class PDE_OP(dde.data.PDEOperatorCartesianProd):
    def __init__(self, 
                pde, 
                function_space, 
                evaluation_points,
                n_pca, 
                num_function, 
                function_variables=None, 
                num_test=None, 
                batch_size=None,
                scale_factor=1,
                pca_save_path = None):
        
        self.scale_factor = scale_factor
        self.pca = PCA(n_components=n_pca)
        self.pca_save_path = pca_save_path
        self.branchmin = None
        self.branchmax = None

        super().__init__(pde, function_space, evaluation_points, num_function, 
                 function_variables, num_test, batch_size) 

        
    def trunk_transform(self,x):  # this function mechod is specificly for 1d
        return torch.cat((x,torch.sqrt(1 - torch.sum(x**2, dim=1, keepdims=True)) ), dim=1)
    
    def branch_transform(self, x, min, max):
        d = torch.tensor(x.shape[1])
        self.branchmax = torch.ones(d) * min
        self.branchmin = torch.ones(d) * max
        x = 2 * (x - self.branchmin) / (self.branchmax - self.branchmin) - 1  # Rescale to [-1, 1]
        x = x / torch.sqrt(d)
        x_d1 = torch.sqrt(1 - torch.sum(x**2, dim=1, keepdims=True))
        return torch.cat((x, x_d1), dim=1)
    
    def train_next_batch(self, batch_size=None):
        if self.train_x is None:

            func_feats = self.func_space.random(self.num_func)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)

            vx = self.scale_factor * self.func_space.eval_batch(
                func_feats, self.pde.train_x[:, self.func_vars]
            )

            # data preprocessing
            self.pca.fit(func_vals)
            joblib.dump(self.pca, self.pca_save_path) # save the pca model 
            print('model saved')
            func_vals = self.pca.transform(func_vals)
            self.train_x = (func_vals, self.pde.train_x)
            self.train_aux_vars = vx

        if self.batch_size is None:
            return self.train_x, self.train_y, self.train_aux_vars

        indices = self.train_sampler.get_next(self.batch_size)
        traix_x = (self.train_x[0][indices], self.train_x[1])
        return traix_x, self.train_y, self.train_aux_vars[indices]

    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
            self.test_aux_vars = self.train_aux_vars
        else:
            func_feats = self.func_space.random(self.num_test)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            vx = self.scale_factor * self.func_space.eval_batch(
                func_feats, self.pde.test_x[:, self.func_vars]
            )
            func_vals = self.pca.transform(func_vals)
            self.test_x = (func_vals, self.pde.test_x)
            self.test_aux_vars = vx
            
        return self.test_x, self.test_y, self.test_aux_vars
    
    def _losses(self, outputs, loss_fn, inputs, model, num_func):
        # PDE
        f = []
        if self.pde.pde is not None:
            f = self.pde.pde(model.zcs_parameters, outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        error_f = [fi[:, bcs_start[-1] :] for fi in f]
        losses = [loss_fn(bkd.zeros_like(error), error) for error in error_f]  # noqa

        # BC
        for k, bc in enumerate(self.pde.bcs):
            beg, end = bcs_start[k], bcs_start[k + 1]
            error_k = []
            # NOTE: this loop over functions can also be avoided if we implement collective ic/bc
            for i in range(num_func):
                output_i = outputs[i]
                if bkd.ndim(output_i) == 1:  # noqa
                    output_i = output_i[:, None]
                error_ki = bc.error(
                    self.train_x[1],
                    inputs[1],
                    output_i,
                    beg,
                    end,
                    aux_var=model.net.auxiliary_vars[i][:, None],
                )
                error_k.append(error_ki)
            error_k = bkd.stack(error_k, axis=0)  # noqa
            loss_k = loss_fn(bkd.zeros_like(error_k), error_k)  # noqa
            losses.append(loss_k)
        return losses


class Model(dde.Model):
    def __init__(self, data, net):
        super().__init__(data, net)
        # store ZCS parameters, sent to user for PDE calculation
        self.zcs_parameters = None

    def _compile_pytorch(self, lr, loss_fn, decay):
        """pytorch"""
        super()._compile_pytorch(lr, loss_fn, decay)

        def process_inputs_zcs(inputs):
            # get inputs
            branch_inputs, trunk_inputs = inputs

            # convert to tensors with grad disabled
            branch_inputs = torch.as_tensor(branch_inputs)
            trunk_inputs = torch.as_tensor(trunk_inputs)

            # create ZCS scalars
            n_dim_crds = trunk_inputs.shape[1]
            zcs_scalars = [
                torch.as_tensor(0.0).requires_grad_() for _ in range(n_dim_crds)
            ]

            # add ZCS to truck inputs
            zcs_vector = torch.stack(zcs_scalars)
            trunk_inputs = trunk_inputs + zcs_vector[None, :]

            # return inputs and ZCS scalars
            return (branch_inputs, trunk_inputs), {"leaves": zcs_scalars}

        def outputs_losses_zcs(training, inputs, targets, auxiliary_vars, losses_fn):
            # aux
            self.net.auxiliary_vars = None
            if auxiliary_vars is not None:
                self.net.auxiliary_vars = torch.as_tensor(auxiliary_vars)

            # inputs
            inputs, self.zcs_parameters = process_inputs_zcs(inputs)

            # forward
            self.net.train(mode=training)
            outputs_ = self.net(inputs)

            # losses
            if targets is not None:
                targets = torch.as_tensor(targets)
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            losses = torch.stack(losses)

            # weighted
            if self.loss_weights is not None:
                losses *= torch.as_tensor(self.loss_weights)

            # clear cached gradients (actually not used with ZCS)
            dde.grad.clear()
            return outputs_, losses

        def outputs_losses_train_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        def outputs_losses_test_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        def train_step_zcs(inputs, targets, auxiliary_vars):
            def closure():
                losses = outputs_losses_train_zcs(inputs, targets, auxiliary_vars)[1]
                # weight = torch.tensor([0.1,1])
                total_loss = torch.sum(losses)
                self.opt.zero_grad()
                total_loss.backward()
                return total_loss

            self.opt.step(closure)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # overwrite callables
        self.outputs_losses_train = outputs_losses_train_zcs
        self.outputs_losses_test = outputs_losses_test_zcs
        self.train_step = train_step_zcs


class LazyGrad:
    """Gradients for ZCS with lazy evaluation."""

    def __init__(self, zcs_parameters, u):
        self.zcs_parameters = zcs_parameters
        self.n_dims = len(zcs_parameters["leaves"])
        self.a = torch.ones_like(u).requires_grad_()
        self.a.stop_gradient = False

        omega = (u * self.a).sum()

        # cached lower-order derivatives of omega
        self.cached_omega_grads = {
            # the only initial element is omega itself, with all orders being zero
            (0,)
            * self.n_dims: omega
        }

    def grad_wrt_z(self, y, z):
        return torch.autograd.grad(y, z, create_graph=True)[0]


    def grad_wrt_a(self, y):
        return torch.autograd.grad(y, self.a, create_graph=True)[0]


    def compute(self, required_orders: Tuple[int, ...]):
        if required_orders in self.cached_omega_grads.keys():
            # derivative w.r.t. a
            return self.grad_wrt_a(self.cached_omega_grads[required_orders])

        # find the start
        orders = np.array(required_orders)
        exists = np.array(list(self.cached_omega_grads.keys()))
        diffs = orders[None, :] - exists
        # existing orders no greater than target element-wise
        avail_indices = np.where(diffs.min(axis=1) >= 0)[0]
        # start from the closet
        start_index = np.argmin(diffs[avail_indices].sum(axis=1))
        start_orders = exists[avail_indices][start_index]

        # dim loop
        for i, zi in enumerate(self.zcs_parameters["leaves"]):
            # order loop
            while start_orders[i] != required_orders[i]:
                omega_grad = self.grad_wrt_z(
                    self.cached_omega_grads[tuple(start_orders)], zi
                )
                start_orders[i] += 1
                self.cached_omega_grads[tuple(start_orders)] = omega_grad

        # derivative w.r.t. a
        return self.grad_wrt_a(self.cached_omega_grads[required_orders])