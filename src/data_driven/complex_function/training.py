import os
os.environ['DDE_BACKEND'] = 'pytorch'

import deepxde as dde
import numpy as np
from deepxde.backend import torch

from ...classical_orthogonal_NN import OrthoNN

#complex example
def func(x):
    return np.sin(x)+np.sin(2*x)+np.sin(3*x)+np.sin(4*x)

def  input_transform(x): #for 1-dimention input
    d = x.shape[1]
    # x_min = torch.min(x,0).values
    # x_max = torch.max(x,0).values
    x = 2*(x+torch.pi)/(2*torch.pi)-1 #rescale to [-1,1]
    x_d1 = torch.sqrt(1-torch.sum(x**2,1,keepdim = True)/d)
    return torch.cat((x,x_d1),dim=1) 


geom = dde.geometry.Interval(-np.pi, np.pi)
num_train = 200
num_test = 100
data = dde.data.Function(geom, func, num_train, num_test)
net =  OrthoNN([2,10,10,10,1],'relu')
net.apply_feature_transform(input_transform)
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000,disregard_previous_best=True)
dde_model = model.net
model.save(r'model_checkpoint')
for name,param in dde_model.named_parameters():
    np.savetxt(fr'/classical_training/{name}.txt',param.cpu().detach().numpy())