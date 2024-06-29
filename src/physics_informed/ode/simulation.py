import sys
import numpy as np
sys.path.append('/mnt/crucial/pengpengxiao/workplace')
from quantum_layer_infinite import tomo_output
from qiskit.providers.aer import AerSimulator
import joblib

branch_hidden0_bias = np.loadtxt(r'classical_training/branch.hidden_layers.0.bias.txt')
branch_hidden0_thetas = np.loadtxt(r'classical_training/branch.hidden_layers.0.thetas.txt')
branch_hidden1_bias = np.loadtxt(r'classical_training/branch.hidden_layers.1.bias.txt')
branch_hidden1_thetas = np.loadtxt(r'classical_training/branch.hidden_layers.1.thetas.txt')
branch_hidden2_bias = np.loadtxt(r'classical_training/branch.hidden_layers.2.bias.txt')
branch_hidden2_thetas = np.loadtxt(r'classical_training/branch.hidden_layers.2.thetas.txt')
branch_hidden3_bias = np.loadtxt(r'classical_training/branch.hidden_layers.3.bias.txt')
branch_hidden3_thetas = np.loadtxt(r'classical_training/branch.hidden_layers.3.thetas.txt')
branch_output_bias = np.loadtxt(r'classical_training/branch.output_layer.bias.txt')
branch_output_weight = np.loadtxt(r'classical_training/branch.output_layer.weight.txt')

trunk_hidden0_bias = np.loadtxt(r'classical_training/trunk.hidden_layers.0.bias.txt')
trunk_hidden0_thetas = np.loadtxt(r'classical_training/trunk.hidden_layers.0.thetas.txt')
trunk_hidden1_bias = np.loadtxt(r'classical_training/trunk.hidden_layers.1.bias.txt')
trunk_hidden1_thetas = np.loadtxt(r'classical_training/trunk.hidden_layers.1.thetas.txt')
trunk_hidden2_bias = np.loadtxt(r'classical_training/trunk.hidden_layers.2.bias.txt')
trunk_hidden2_thetas = np.loadtxt(r'classical_training/trunk.hidden_layers.2.thetas.txt')
trunk_hidden3_bias = np.loadtxt(r'classical_training/trunk.hidden_layers.3.bias.txt')
trunk_hidden3_thetas = np.loadtxt(r'classical_training/trunk.hidden_layers.3.thetas.txt')
trunk_output_bias = np.loadtxt(r'classical_training/trunk.output_layer.bias.txt')
trunk_output_weight = np.loadtxt(r'classical_training/trunk.output_layer.weight.txt')

b = np.loadtxt(r'classical_training/b.txt')

simulator = AerSimulator(device='GPU')

def tanh(x):
    return np.tanh(x)

d2 = np.load(r'picked_aligned_test.npz',allow_pickle=True)
x_test0,y_test0 = (d2['X0'].astype(np.float32),d2['X1'].astype(np.float32)),d2['y'].astype(np.float32)

def trunk_transform(x):  
    return np.concatenate((x,np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True)) ), axis=1)

def branch_transform(x):
    d = x.shape[1]
    branchmax = np.ones(d) * 40
    branchmin = -np.ones(d) * 40
    x = 2 * (x - branchmin) / (branchmax - branchmin) - 1  # Rescale to [-1, 1]
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    x = x / np.sqrt(d)
    x = np.concatenate((x, x_d1), axis=1)
    return x

loaded_pca = joblib.load('classical_training/pca_model.joblib')

x_test = (branch_transform(loaded_pca.transform(x_test0[0])),trunk_transform(x_test0[1]))

branch_outputs, trunk_outputs = [], []  
for x_branch0 in x_test[0]:
    x_branch = x_branch0
    x_branch = tomo_output(10+1,20, x_branch, branch_hidden0_thetas,simulator)+branch_hidden0_bias
    x_branch = tanh(x_branch)
    
    x_branch = tomo_output(20,20, x_branch, branch_hidden1_thetas,simulator)+branch_hidden1_bias
    x_branch = tanh(x_branch)
    
    x_branch = tomo_output(20,20, x_branch, branch_hidden2_thetas,simulator)+branch_hidden2_bias
    x_branch = tanh(x_branch)
    
    x_branch = tomo_output(20,20, x_branch, branch_hidden3_thetas,simulator)+branch_hidden3_bias
    x_branch = tanh(x_branch)

    x_branch = np.dot(x_branch,branch_output_weight.T) +branch_output_bias

    branch_outputs.append(x_branch.copy())


for x_trunk in x_test[1]:
    #trank
    x_trunk = tomo_output(1+1,20, x_trunk, trunk_hidden0_thetas,simulator)+trunk_hidden0_bias
    x_trunk = tanh(x_trunk)
    
    x_trunk = tomo_output(20,20, x_trunk, trunk_hidden1_thetas,simulator)+trunk_hidden1_bias
    x_trunk = tanh(x_trunk)
    
    x_trunk = tomo_output(20,20, x_trunk, trunk_hidden2_thetas,simulator)+trunk_hidden2_bias
    x_trunk = tanh(x_trunk)
    
    x_trunk = tomo_output(20,20, x_trunk, trunk_hidden3_thetas,simulator)+trunk_hidden3_bias
    x_trunk = tanh(x_trunk)
    
    x_trunk = np.dot(x_trunk,trunk_output_weight.T) +trunk_output_bias
    x_trunk = tanh(x_trunk)

    trunk_outputs.append(x_trunk.copy())

x = np.einsum('bi,ni->bn',np.array(branch_outputs),np.array(trunk_outputs))+b
x = x * x_test0[1].reshape(1,-1)


mean_l2_error = np.mean(np.linalg.norm(x-y_test0,axis=1)/np.linalg.norm(y_test0,axis=1))

np.savetxt(r'simulation/mean_l2_error.txt',[mean_l2_error])

