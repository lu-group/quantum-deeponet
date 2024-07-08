import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel, depolarizing_error
from ...quantum_layer_noise import tomo_output




# simple function N2_[2,3,3,1],'tanh',80,50000,264.8s,\math-container{0.149%}
hidden_layers0_thetas = np.loadtxt(r'/classical_training/hidden_layers.0.thetas.txt')
hidden_layers0_bias = np.loadtxt(r'/classical_training/hidden_layers.0.bias.txt')
hidden_layers1_thetas = np.loadtxt(r'/classical_training/hidden_layers.1.thetas.txt')
hidden_layers1_bias = np.loadtxt(r'/classical_training/hidden_layers.1.bias.txt')
output_layer_bias = np.loadtxt(r'/classical_training/output_layer.bias.txt')
output_layer_weight = np.loadtxt(r'/classical_training/output_layer.weight.txt')


def tanh(x):
    return np.tanh(x)

def  input_transform(x): #for 1-dimention input
    x_d1= np.sqrt(1-x**2)
    return np.array([x,x_d1])

# construct network
def simple_function_NN(input,shots):
    x = input
    x = input_transform(x)
    x = tomo_output(2,3, x, hidden_layers0_thetas, noise_sim,shots)+hidden_layers0_bias
    x = tanh(x)
    x = tomo_output(3,3, x, hidden_layers1_thetas, noise_sim,shots)+hidden_layers1_bias
    x = tanh(x)
    x = np.dot(x,output_layer_weight.T) +output_layer_bias
    return x

params = np.linspace(0.0, 0.1, 20)
for i in range(3):
    l2_errors,y_arrays = [],[]
    for param in params:

        provider = IBMProvider(instance = 'ibm-q/open/main')
        noise_model = NoiseModel(basis_gates=['ecr', 'id', 'rz', 'sx', 'x'])
        error_all_qubit = depolarizing_error(param,1)
        error_all_qubit2 = depolarizing_error(0.8*param, 2) 
        noise_model.add_all_qubit_quantum_error(error_all_qubit, ['id', 'rz', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(error_all_qubit2, ['ecr'])

        noise_sim = AerSimulator(device='GPU',noise_model=noise_model)

        # import pdb; pdb.set_trace()

        x_array = np.linspace(-1,1,100)
        y_true = 1/(1+25*x_array**2)
        
        y_array = []
        for x in x_array:
            y = simple_function_NN(x,10000000)
            y_array.append(y)
        l2_error = np.linalg.norm(y_array-y_true)/np.linalg.norm(y_true)
        l2_errors.append(l2_error)
        y_arrays.append(y_array)
    np.savez(f'depolarizing_{i}',y_arrays=y_arrays,params = params)
    np.savetxt(f'depolarizating_error_{i}.txt',l2_errors)
