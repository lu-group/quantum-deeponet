import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def RBS(theta): # RBS gate with parameter t
    rbs_q = QuantumRegister(2)
    c_qubit = rbs_q[0]
    t_qubit = rbs_q[1]
    rbs = QuantumCircuit([c_qubit,t_qubit], name='RBS_'+str(theta))
    rbs.h(c_qubit)
    rbs.h(t_qubit)
    rbs.cz(c_qubit, t_qubit)

    rbs.ry(theta, c_qubit)
    rbs.ry(-theta, t_qubit)

    rbs.cz(c_qubit, t_qubit)
    rbs.h(c_qubit)
    rbs.h(t_qubit) 
    return rbs.to_gate()

def data_loader(data_array): # NOTE: initial state is |10000...> instead of ground state
    # data_array: an array of input data such that ||data_array||_2 = 1
    # If data is not normalized, then it will do that
    # Check Section 2.3 in paper https://arxiv.org/abs/2106.07198
    if len(data_array) < 2:
        raise Exception("Data size must be at least 2")
        
    # Normalize data if necessary
    data_norm = np.linalg.norm(data_array, ord=2)
    if np.abs(data_norm - 1) > 1e-8:
#         warnings.warn("Normalizing data")
        data_array = data_array/data_norm
        
    # Compute parameters
    num_qubits = len(data_array)
    num_params = num_qubits - 1
    
    sinm1_value_inside_acos = 1 # to record sin^-1(alpha_0)sin^-1(alpha_1)...
    params = []
    for i in range(0, num_params):
#         print(data_array[i] * sinm1_value_inside_acos)
        params.append(np.arccos(data_array[i] * sinm1_value_inside_acos))
        sinm1_value_inside_acos = sinm1_value_inside_acos * 1/np.sin(params[i])
    params[-1] = np.arctan(data_array[-1]/data_array[-2])
    if data_array[-1] < 0 and data_array[-2] < 0: # we only need num_params data to determine params
        params[-1] = params[-1]-np.pi
    if data_array[-1] > 0 and data_array[-2] < 0: 
        params[-1] = params[-1]+np.pi
    
    loader_qr = QuantumRegister(num_qubits)
    loader_circuit = QuantumCircuit(loader_qr)
    for i in range(num_qubits-1):
#         loader_circuit.compose(RBS(params[i]), qubits=[loader_qr[num_qubits-i-1], loader_qr[num_qubits-i-2]], inplace=True)
        loader_circuit.compose(RBS(params[i]), qubits=[i,i+1], inplace=True)

    # fig = loader_circuit.draw(output='mpl')
    # display(fig)
    return loader_circuit.to_gate()
                                                  
def find_nparams(n,d): # size_in, size_out
    return int((2*n-1-d)*(d/2))

def W(n_in, n_out, thetas): #generate thetas else where
    
    larger_features = max(n_in,n_out)
    smaller_features = min(n_in,n_out)

    correct_size = int((2*larger_features - 1 - smaller_features) * smaller_features * 0.5)
    if len(thetas) != correct_size:
        raise Exception("Size of parameter should be {:d} but now it is {:d}".format(correct_size, len(thetas)))
    
    W_qr = QuantumRegister(larger_features)
    W_circuit = QuantumCircuit(W_qr)

    if larger_features == smaller_features:
        smaller_features -= 1 #6-6 6-5 have the same pyramid
    q_end_indices = np.concatenate([
        np.arange(2, larger_features +1 ),
        larger_features + 1 - np.arange(2, smaller_features +1 )
    ]) 
    q_start_indices = np.concatenate([
        np.arange(q_end_indices.shape[0] + smaller_features - larger_features)%2,# [0, 1, 0, 1, ...]
        np.arange( larger_features- smaller_features)
    ])  

    q_slice_sizes = q_end_indices - q_start_indices

    if n_in <n_out:  # generate the pyramid for in_features < out_features case
        q_end_indices = q_end_indices[::-1]
        q_start_indices = q_start_indices[::-1]
        q_slice_sizes =  q_slice_sizes[::-1]
        # pad x fist if in_features < out_features case

    theta_start_index = 0

    for i,q_start_index in enumerate(q_start_indices):
        
        theta_slice = thetas[theta_start_index:theta_start_index+q_slice_sizes[i]//2]

        # import pdb; pdb.set_trace()
        for theta in theta_slice:
            #print('theta',theta)
            W_circuit.compose(RBS(theta), qubits=[W_qr[q_start_index], W_qr[ q_start_index+1]], inplace=True)
            q_start_index += 2
        theta_start_index += q_slice_sizes[i]//2
    # fig = W_circuit.draw(output='mpl')
    # plt.show()
    return W_circuit.to_gate()

def custom_tomo(n_in, n_out, data_array, thetas):
     #len(data_array) should be equal to n_in

    num_qubits = max(n_in,n_out)
    special_arr = np.array([1/np.sqrt(num_qubits)]*num_qubits)
    
    anc_qr = QuantumRegister(1)
    anc_cr = ClassicalRegister(1)
    tomo_qr = QuantumRegister(num_qubits) #construct a larger circuit
    tomo_cr = ClassicalRegister(n_out)
    tomo_circuit = QuantumCircuit(anc_qr, tomo_qr, anc_cr, tomo_cr)
  

    input_qubits = [i for i in range(num_qubits-n_in+1,num_qubits+1)] # put dataloader at the bottom of the pyramid
    tomo_qubits = [i for i in range(1,num_qubits+1)]
    
    tomo_circuit.h(anc_qr)
    tomo_circuit.cx(anc_qr, tomo_qr[num_qubits-n_in])
    tomo_circuit.compose(data_loader(data_array), qubits=input_qubits, inplace=True)
    tomo_circuit.compose(W(n_in, n_out, thetas), qubits=tomo_qr, inplace=True)
    tomo_circuit.compose(data_loader(special_arr).inverse(), qubits=tomo_qubits, inplace=True)
    tomo_circuit.barrier()
    
    tomo_circuit.x(anc_qr)
    tomo_circuit.cx(anc_qr, tomo_qr[0])
    tomo_circuit.compose(data_loader(special_arr), qubits=tomo_qubits, inplace=True)
    tomo_circuit.barrier()
    
    tomo_circuit.h(anc_qr)

    # fig = tomo_circuit.draw(output='mpl')
    # display(fig)

    return tomo_circuit