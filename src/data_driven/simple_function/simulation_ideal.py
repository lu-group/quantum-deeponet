import numpy as np
from qiskit_aer import AerSimulator
from ...quantum_layer_ideal import tomo_output

# simple function N2_[2,3,3,1],'tanh',80,50000,264.8s,\math-container{0.149%}
hidden_layers0_thetas = np.loadtxt(r'/classical_training/hidden_layers.0.thetas.txt')
hidden_layers0_bias = np.loadtxt(r'/classical_training/hidden_layers.0.bias.txt')
hidden_layers1_thetas = np.loadtxt(r'/classical_training/hidden_layers.1.thetas.txt')
hidden_layers1_bias = np.loadtxt(r'/classical_training/hidden_layers.1.bias.txt')
output_layer_bias = np.loadtxt(r'/classical_training/output_layer.bias.txt')
output_layer_weight = np.loadtxt(r'/classical_training/output_layer.weight.txt')


simulator_gpu = AerSimulator(device='GPU')

def tanh(x):
    return np.tanh(x)

def  input_transform(x): #for 1-dimention input
    x_d1= np.sqrt(1-x**2)
    return np.array([x,x_d1])

# construct network
def simple_function_NN(input):
    x = input
    x = input_transform(x)
    x = tomo_output(2,3, x, hidden_layers0_thetas, simulator_gpu)+hidden_layers0_bias
    x = tanh(x)
    x = tomo_output(3,3, x, hidden_layers1_thetas, simulator_gpu)+hidden_layers1_bias
    x = tanh(x)
    x = np.dot(x,output_layer_weight.T) +output_layer_bias
    return x


x_array = np.linspace(-1,1,100)
y_true = 1/(1+25*x_array**2)
l2_errors = []
y_array = []
for x in x_array:
    y = simple_function_NN(x)
    y_array.append(y)
l2_error = np.linalg.norm(y_array-y_true)/np.linalg.norm(y_true)
l2_errors.append(l2_error)

np.save('ideal.npy',y_array)
np.savetxt('ideal_error.txt',l2_errors)
