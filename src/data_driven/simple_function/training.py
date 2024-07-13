import numpy as np
import deepxde as dde
import torch
from ...classical_orthogonal_NN import OrthoNN

def func(x):
    return 1/(1+25*x**2)

x = np.linspace(-1,1,100)

def func(x):
    return 1/(1+25*x**2)

def  input_transform(x): #for 1-dimention input
    x_d1 = torch.sqrt(1-torch.sum(x**2,1,keepdim = True))
    return torch.cat((x,x_d1),dim=1) 

geom = dde.geometry.Interval(-1, 1)
num_train = 80
num_test = 100
data = dde.data.Function(geom, func, num_train, num_test)
net =  OrthoNN([2,3,3,1],'tanh')
net.apply_feature_transform(input_transform)
model = dde.Model(data, net)
model.compile("adam", lr=0.0001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000,disregard_previous_best=True)
dde_model = model.net
model.save(r'model_checkpoint')
for name,param in dde_model.named_parameters():
    np.savetxt(fr'/classical_training/{name}.txt',param.cpu().detach().numpy())