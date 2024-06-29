import torch
import numpy as np

class OrthoLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        larger_features = max(in_features,out_features)
        smaller_features = min(in_features,out_features)
        size = (2*larger_features-1-smaller_features)*smaller_features/2 #number of free parameters
        # torch.manual_seed(0)
        self.thetas = torch.nn.Parameter(torch.randn(int(size))) #normal distribution initializer for thetas
        self.bias = torch.nn.Parameter(torch.zeros(int(out_features)))


    def hidden_layer(self, x ,in_features, out_features):
        larger_features = max(in_features,out_features)
        smaller_features = min(in_features,out_features)

        if larger_features == smaller_features:
            smaller_features -= 1 #6-6 6-5 have the same pyramid
        x_end_index = np.concatenate([
            np.arange(2, larger_features +1 ),
            larger_features + 1 - np.arange(2, smaller_features +1 )
        ]) 
        x_start_index = np.concatenate([
            np.arange(x_end_index.shape[0] + smaller_features - larger_features)%2,# [0, 1, 0, 1, ...]
            np.arange( larger_features- smaller_features)
        ])  

        x_slice_sizes = x_end_index - x_start_index

        if in_features < out_features:  # generate the pyramid for in_features < out_features case
            x_end_index = x_end_index[::-1]
            x_start_index = x_start_index[::-1]
            x_slice_sizes =  x_slice_sizes[::-1]
            x = torch.nn.functional.pad(x,(out_features-x.shape[1],0)) # pad x fist if in_features < out_features case

        theta_start_index = 0

        for i in range(len(x_start_index)):
            
            theta_slice = self.thetas[theta_start_index:theta_start_index+x_slice_sizes[i]//2]
            theta_start_index = theta_start_index+x_slice_sizes[i]//2
            x_slice = x[:, x_start_index[i] :x_end_index[i]]

            # generate rotation matrix
            n = len(theta_slice)
            row_indices = torch.cat([torch.tensor([2*i,2*i,2*i+1,2*i+1]) for i in range(n)])
            column_indices = torch.cat([torch.tensor([2*i,2*i+1]).repeat(2) for i in range(n)])
            indices = torch.stack([row_indices,column_indices])
            theta_slice = theta_slice.view(-1,1)
            values = torch.cat([torch.cos(theta_slice),torch.sin(theta_slice),-torch.sin(theta_slice),torch.cos(theta_slice)],dim=1).view(-1)
            rotation_matrix = torch.sparse_coo_tensor(indices,values,size=[2*n,2*n])
            x_new = x.clone()
            x_new[:, x_start_index[i] :x_end_index[i]] = torch.mm(x_slice,rotation_matrix)
            x = x_new #to avoid in-place operation
        
        if in_features > out_features:
            x = x[:,in_features-out_features:]

        return x + self.bias

    def forward(self,x):
        if x.shape[1] != self.in_features:
            raise AssertionError(
                f'x shape {x.shape} isn\'t equal to {self.in_features}'
                )
        x = self.hidden_layer(x,self.in_features,self.out_features)
        return x            