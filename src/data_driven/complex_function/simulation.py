import numpy as np
from quantum_layer_ideal import tomo_output
from qiskit.providers.aer import AerSimulator

# Load parameters

# simple function N2_[2,3,3,1],'tanh',80,50000,264.8s,\math-container{0.149%}
hidden_layers0_thetas = np.loadtxt(r'/classical_training/hidden_layers.0.thetas.txt')
hidden_layers0_bias = np.loadtxt(r'/classical_training/hidden_layers.0.bias.txt')
hidden_layers1_thetas = np.loadtxt(r'/classical_training/hidden_layers.1.thetas.txt')
hidden_layers1_bias = np.loadtxt(r'/classical_training/hidden_layers.1.bias.txt')
hidden_layers2_thetas = np.loadtxt(r'/classical_training/hidden_layers.2.thetas.txt')
hidden_layers2_bias = np.loadtxt(r'/classical_training/hidden_layers.2.bias.txt')
output_layer_bias = np.loadtxt(r'/classical_training/output_layer.bias.txt')
output_layer_weight = np.loadtxt(r'/classical_training/output_layer.weight.txt')


simulator_gpu = AerSimulator(device='GPU')

def relu(x):
    return np.maximum(0,x)

def  input_transform(x): #for 1-dimention input
    x = 2*(x+np.pi)/(2*np.pi)-1
    x_d1= np.sqrt(1-x**2)
    return np.array([x,x_d1])


x_array = np.linspace(-np.pi,np.pi,100)
y0, y1, y2, y3 = [], [], [], []
for x in x_array:
    x = input_transform(x)
    x = tomo_output(2,10, x, hidden_layers0_thetas, simulator_gpu)+hidden_layers0_bias
    x = relu(x)
    y0.append(np.copy(x))
    x = tomo_output(10,10, x, hidden_layers1_thetas, simulator_gpu)+hidden_layers1_bias
    x = relu(x)
    y1.append(np.copy(x))
    x = tomo_output(10,10, x, hidden_layers2_thetas, simulator_gpu)+hidden_layers2_bias
    x = relu(x)
    y2.append(np.copy(x))
    x = np.dot(x,output_layer_weight.T) +output_layer_bias
    y3.append(np.copy(x))


y_true = np.sin(x_array)+np.sin(2*x_array)+np.sin(3*x_array)+np.sin(4*x_array) 

l2_error = np.linalg.norm(y3-y_true)/np.linalg.norm(y_true)
print('L2 relative error for forward pass',l2_error)

# compute l2 relative error
np.savez(r'forward_data.npz',y_true = y_true,y3=y3, y2=y2, y1=y1, y0=y0, x_array=x_array )
np.savetxt(r'l2_relative_error.txt',[l2_error])
