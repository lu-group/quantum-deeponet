import numpy as np
import numpy.linalg as nl
import cvxopt
import qiskit_ibm_provider
from qiskit import transpile
from quantum_layer import custom_tomo   

def find_least_norm(ptilde, nQubits):
    """
    Solve min ||ptilde - p||_2
          s.t.
            each entry of p sums to 1
            each entry of p is non-negative

    Parameters
    ----------
    nQubits : int
        Number of qubits.
    ptilde : array
        probability vector.

    Returns
    -------
    sol['status']: String
        'optimal' if solve successfully.
    sol['x']: array
        the optimizer.

    """
    # Formulation
    Q = 2 * cvxopt.matrix(np.identity(2**nQubits))
    p = -2 * cvxopt.matrix(ptilde)

    G = -cvxopt.matrix(np.identity(2**nQubits))
    h = cvxopt.matrix(np.zeros(2**nQubits))

    A = cvxopt.matrix(np.ones(2**nQubits), (1, 2**nQubits))
    b = cvxopt.matrix(1.0)

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(Q, p, G, h, A, b)
    return sol['status'], sol['x']

def d2v(count_dict):
    num_qubits = len(  list(count_dict.keys())[0]  )
    prob_vec = np.zeros(2**num_qubits, dtype=float)
    total_counts = np.sum( list(count_dict.values())  )
    prob_vec[ [int(i,2) for i in count_dict.keys()] ] = list(count_dict.values())/total_counts
    return prob_vec

class MeasErrFilter():
    def __init__(self, backend):
        # backend should be a IBMQ bckend or a dict with measurement error parameters
        # in the format
        # {'0': [m1p0_0,m0p1_0], '1':[m1p0_1,m0p1_1]} where key is the physical qubit index
        # and m1p0_0 is the Pr(Meas 1 | Prep 0) for Qubit 0
        # and m0p1_0 is the Pr(Meas 0 | Prep 1) for Qubit 0
        self.backend_prop = None
        self.meas_prop = None
        if type(backend) is qiskit_ibm_provider.ibm_backend.IBMBackend:
            self.backend_prop = backend.properties()
        elif type(backend) is dict:
            self.meas_prop = backend
        else:
            raise Exception("Use a valid IBMQ backend or a dictionary in the required format.")

    def filter(self,count_dict, qubit_list):
        noisy_prob_vec = d2v(count_dict)
        nQubits = int(np.log2(len(noisy_prob_vec)))
        # Create measurement matrix
        M = None
        for q in qubit_list:
            if self.backend_prop is not None:
                pm1p0 = self.backend_prop.qubit_property(q,'prob_meas1_prep0')[0]
                pm0p1 = self.backend_prop.qubit_property(q,'prob_meas0_prep1')[0]
            else:
                pm1p0 = self.meas_prop[str(q)][0]
                pm0p1 = self.meas_prop[str(q)][1]
            if M is None:
                # M = np.array([[1-pm1p0, pm1p0], [pm0p1, 1-pm0p1]])
                M = np.array([[1-pm1p0, pm0p1], [pm1p0, 1-pm0p1]])
            else:
                Msub = np.array([[1-pm1p0, pm0p1], [pm1p0, 1-pm0p1]])
                # M = np.kron(M, Msub)
                M = np.kron(Msub, M)
        rough_denoised_vec = nl.solve(M, noisy_prob_vec)
        # check if it is a valid prob vec
        valid = True
        if not np.allclose(np.sum(rough_denoised_vec), 1.0):
            valid = False
        if np.sum(rough_denoised_vec >= 0.0) != len(rough_denoised_vec):
            valid = False
        if not valid:
            # Solve for noise-mitigated vector
            proc_status, proc_vec = find_least_norm(
                        nl.solve(M, noisy_prob_vec), nQubits)
            if proc_status != 'optimal':
                raise Exception('Sorry, filtering has failed')
            denoised_vec = np.array(proc_vec).flatten()
        else:
            denoised_vec = rough_denoised_vec
        return denoised_vec
    
    def apply(self,prob_vec, qubit_list):
        # Create measurement matrix
        M = None
        for q in qubit_list:
            if self.backend_prop is not None:
                pm1p0 = self.backend_prop.qubit_property(q,'prob_meas1_prep0')
                pm0p1 = self.backend_prop.qubit_property(q,'prob_meas0_prep1')
            else:
                pm1p0 = self.meas_prop[str(q)][0]
                pm0p1 = self.meas_prop[str(q)][1]
            if M is None:
                # M = np.array([[1-pm1p0, pm1p0], [pm0p1, 1-pm0p1]])
                M = np.array([[1-pm1p0, pm0p1], [pm1p0, 1-pm0p1]])
            else:
                Msub = np.array([[1-pm1p0, pm0p1], [pm1p0, 1-pm0p1]])
                # M = np.kron(M, Msub)
                M = np.kron(Msub, M)
        return M.dot(prob_vec)

# qunatum layer with noise
def tomo_output(n_in, n_out, data_array, thetas,simulator, shots, qubit_mapping = None, meas_err_datadict= None): #* meas_err_datadict means including measurement error by ourselves
    
    if (meas_err_datadict is not None) and (qubit_mapping is None):
        raise Exception('for adding measurement, please input the qubit mapping')
    
    for i in range(len(data_array)):
        if np.abs(data_array[i]) < 1e-7:
            data_array[i] += 1e-7
    tomo_circuit = custom_tomo(n_in, n_out, data_array, thetas)
    tomo_circuit.save_density_matrix()
    if qubit_mapping is None:
        state = simulator.run(transpile(tomo_circuit, simulator, seed_transpiler=7, optimization_level=0),shots=shots,target_gpus=[1]).result()
        # state = simulator.run(transpile(tomo_circuit, simulator, seed_transpiler=7, optimization_level=0),shots=shots).result()
    else:
        state = simulator.run(transpile(tomo_circuit, simulator, initial_layout=qubit_mapping, seed_transpiler=7, optimization_level=0),shots=shots,target_gpus=[0]).result()
    
    result = state.data()['density_matrix'].data
    prob = result.diagonal().real
    probcopy = prob.copy()
    probcopy[probcopy<0] = 0
    prob = probcopy/np.sum(probcopy)

    if meas_err_datadict is not None:
        prob = MeasErrFilter(meas_err_datadict).apply(prob,qubit_mapping)

    measurements =  np.random.choice(np.arange(len(prob)),size=shots, p=prob)
    elements , counts = np.unique(measurements, return_counts=True)
    all_sampled = np.zeros(len(prob))
    all_sampled[elements] = counts

    # error mitigation
    non_zero_index = []
    for i in range(np.maximum(n_in,n_out)):
        pos = ['0']*np.maximum(n_in,n_out)
        pos[i] = '1'
        pos0 = ['0']+pos
        pos1 = ['1']+pos
        pos0 = ''.join(pos0)[::-1]
        pos1 = ''.join(pos1)[::-1]
        non_zero_index.extend([int(pos0,2),int(pos1,2)])

    picked_sample = np.zeros(len(prob))
    picked_sample[non_zero_index] = all_sampled[non_zero_index]
    new_count = np.sum(picked_sample)
    
    picked_sample = picked_sample/new_count
    # print(picked_sample)

    output = []
    for i in range(n_out):
        pos = ['0']*n_out
        pos[i] = '1'
        pos0 = ['0']+['0']*(n_in-n_out)+pos
        pos1 = ['1']+['0']*(n_in-n_out)+pos
        pos0 = ''.join(pos0)[::-1]
        pos1 = ''.join(pos1)[::-1]
        result0 = picked_sample[int(pos0,2)]
        result1 = picked_sample[int(pos1,2)]
        output.append(np.sqrt(np.maximum(n_in,n_out))*(result0-result1))
    
    return output