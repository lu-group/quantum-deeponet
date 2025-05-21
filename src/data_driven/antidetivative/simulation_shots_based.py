import numpy as np
from qiskit.providers.aer import AerSimulator
from ...quantum_layer_shots import tomo_output

branch_hidden0_bias = np.loadtxt(r'/classical_training/branch.hidden_layers.0.bias.txt')
branch_hidden0_thetas = np.loadtxt(r'/classical_training/branch.hidden_layers.0.thetas.txt')
branch_output_bias = np.loadtxt(r'/classical_training/branch.output_layer.bias.txt')
branch_output_weight = np.loadtxt(r'/classical_training/branch.output_layer.weight.txt')

trunk_hidden0_bias = np.loadtxt(r'/classical_training/trunk.hidden_layers.0.bias.txt')
trunk_hidden0_thetas = np.loadtxt(r'/classical_training/trunk.hidden_layers.0.thetas.txt')
trunk_output_bias = np.loadtxt(r'/classical_training/trunk.output_layer.bias.txt')
trunk_output_weight = np.loadtxt(r'/classical_training/trunk.output_layer.weight.txt')
b = np.loadtxt(r'/classical_training/b.txt')

simulator = AerSimulator(device='GPU')

def relu(x):
    return np.maximum(0,x)

#load test data, make sure we use the same input transformation as training
d1 = np.load(r'picked_aligned_train.npz',allow_pickle=True)
x_train,y_train = (d1['X0'].astype(np.float32),d1['X1'].astype(np.float32)),d1['y'].astype(np.float32)
d2 = np.load(r'picked_aligned_test.npz',allow_pickle=True)
x_test,y_test = (d2['X0'].astype(np.float32),d2['X1'].astype(np.float32)),d2['y'].astype(np.float32)

trunk_min = np.min( np.stack((np.min(x_train[1], axis=0),np.min(x_test[1], axis=0)),axis=0),axis=0)
trunk_max = np.max( np.stack((np.max(x_train[1], axis=0),np.max(x_test[1], axis=0)),axis=0),axis=0)
branch_min = np.min(np.stack((np.min(x_train[0], axis=0),np.min(x_test[0], axis=0)),axis=0),axis=0)
branch_max = np.max( np.stack((np.max(x_train[0], axis=0),np.max(x_test[0], axis=0)),axis=0),axis=0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
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


x_train = (branch_transform(x_train[0]),trunk_transform(x_train[1]))
x_test = (branch_transform(x_test[0]),trunk_transform(x_test[1]))

for i in range(3):
    branch0_array, branch1_array, trunk0_array, trunk1_array, output, errors = [],[],[],[],[],[]
    for shots in [1000,10000,100000,1000000,10000000,100000000]:

        branch0, branch1, trunk0, trunk1 = [],[],[],[]
        for x_branch0 in x_test[0]:
            x_branch = x_branch0
            x_branch = tomo_output(10+1,10, x_branch, branch_hidden0_thetas,simulator,shots)+branch_hidden0_bias
            x_branch = relu(x_branch)

            branch0.append(x_branch)

            x_branch = np.dot(x_branch,branch_output_weight.T) +branch_output_bias

            branch1.append(x_branch)


        for x_trunk in x_test[1]:
            #trank
            x_trunk = tomo_output(1+1,10, x_trunk, trunk_hidden0_thetas,simulator,shots)+trunk_hidden0_bias
            x_trunk = relu(x_trunk)

            trunk0.append(x_trunk)

            x_trunk = np.dot(x_trunk,trunk_output_weight.T) +trunk_output_bias
            x_trunk = relu(x_trunk)

            trunk1.append(x_trunk)

        x = np.einsum('bi,ni->bn',np.array(branch1),np.array(trunk1))+b
        mean_l2_error = np.mean(np.linalg.norm(x-y_test,axis=1)/np.linalg.norm(y_test,axis=1))
        
        branch0_array.append(branch0)
        branch1_array.append(branch1)
        trunk0_array.append(trunk0)
        trunk1_array.append(trunk1)
        output.append(x)
        errors.append(mean_l2_error)
    

    np.savetxt(fr'mean_l2_error{i}.txt',errors)
    np.savez(fr'forward_data{i}.npz',
            branch0_array=branch0_array,branch1_array=branch0_array,
            trunk0_array = trunk0_array,trunk1_array=trunk1_array,output=output)