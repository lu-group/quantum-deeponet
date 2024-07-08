import numpy as np
from qiskit import transpile
from quantum_layer_ideal import custom_tomo

# quantum layer without noise
def tomo_output(n_in, n_out, data_array, thetas, simulator, shots,):
    for i in range(len(data_array)):
        if np.abs(data_array[i]) < 1e-7:
            data_array[i] += 1e-7
    tomo_circuit = custom_tomo(n_in, n_out, data_array, thetas)
    tomo_circuit.save_statevector('state')
    state = simulator.run(transpile(tomo_circuit, simulator),shots=1,target_gpus=[1]).result()
    result = np.real(state.data()['state'].data)
    prob = result**2
    
    measurements =  np.random.choice(np.arange(len(prob)),size=shots, p=prob)
    elements , counts = np.unique(measurements, return_counts=True)
    sampled = counts/shots
    all_sampled = np.zeros(len(prob))
    all_sampled[elements] = sampled

    output = []
    for i in range(n_out):
        pos = ['0']*n_out
        pos[i] = '1'
        pos0 = ['0']+['0']*(n_in-n_out)+pos
        pos1 = ['1']+['0']*(n_in-n_out)+pos
        pos0 = ''.join(pos0)[::-1]
        pos1 = ''.join(pos1)[::-1]
        result0 = all_sampled[int(pos0,2)]
        result1 = all_sampled[int(pos1,2)]
        output.append(np.sqrt(np.maximum(n_in,n_out))*(result0-result1))
    
    return output