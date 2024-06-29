import os
os.environ['DDE_BACKEND'] = 'pytorch'
print(os.environ.get('DDE_BACKEND'))
import deepxde as dde
from deepxde.data.function_spaces import GRF
import numpy as np
import matplotlib.pyplot as plt

# here gen_data_GRF can generate u and v with respect to different x 

# solve pde using finite difference method
def compute_numerical_solution(x, v, N):
    h = x[1]-x[0]
    #Kx=b
    K = 1 * np.eye(N-1) - np.eye(N-1,k=-1)
    b = h* v[1:]
    u = np.linalg.solve(K,b)
    return np.concatenate(([0], u))

def gen_data_GRF(M, Nv,Nu, a=1,l=1,train = 150, test = 1000):
    # generate v and compute u in a dense grid with size M
    # x in [0,1] GRF.x.shape = [1, M]
    # l: lenth scale, a is the amplitude
    space = GRF(1, kernel = 'RBF', length_scale = l, N = M, interp = 'cubic')
    x = np.ravel(space.x)
    v_full = []
    u_full = []

    for _ in range(train):
        v = np.ravel(a * space.random(1)) # space.random(1) shape = [1, M], v.shape = [M,1]
        u = compute_numerical_solution(x,v,M) #u.shape = [M]
        v_full.append(v)
        u_full.append(u)
        # plt.figure()
        # plt.plot(x,v,label='v')
        # plt.plot(x,u,label='u')
        # plt.legned()
        # plt.show()

    v_full = np.array(v_full,dtype=np.float32)
    u_full = np.array(u_full,dtype=np.float32)
    np.savez_compressed('data_ode_simple/full_aligned_train.npz',X0 = v_full, X1 = x.reshape(-1,1) , y = u_full)
     
    # down sampling and take N points
    index_v = [round(M/Nv)*i for i in range(Nv-1)]+[M-1]
    index_u = [round(M/Nu)*i for i in range(Nu-1)]+[M-1]
    
    v = v_full[:,index_v]
    xv = x[index_v]
    xu = x[index_u]
    u = u_full[:,index_u]

    np.savez_compressed('data_ode_simple/picked_aligned_train.npz',X0 = v, X1 = xu.reshape(-1,1) , y = u,  X0_p = xv.reshape(-1,1))

    #test data
    x = np.ravel(space.x)
    v_full = []
    u_full = []
    for _ in range(test):
        v = np.ravel(a * space.random(1)) # space.random(1) shape = [1, M], v.shape = [M,1]
        u = compute_numerical_solution(x,v,M) #u.shape = [M]
        v_full.append(v)
        u_full.append(u)
        # plt.figure()
        # plt.plot(x,v,label='v')
        # plt.plot(x,u,label='u')
        # plt.legned()
        # plt.show()

    v_full = np.array(v_full,dtype=np.float32)
    u_full = np.array(u_full,dtype=np.float32)
    np.savez_compressed('data_ode_simple/full_aligned_test.npz',X0 = v_full, X1 = x.reshape(-1,1) , y = u_full)
     
    # down sampling and take N points
    index_v = [round(M/Nv)*i for i in range(Nv-1)]+[M-1]
    index_u = [round(M/Nu)*i for i in range(Nu-1)]+[M-1]
    
    v = v_full[:,index_v]
    xv = x[index_v]
    xu = x[index_u]
    u = u_full[:,index_u]


    np.savez_compressed('data_ode_simple/picked_aligned_test.npz',X0 = v, X1 = xu.reshape(-1,1) , y = u,  X0_p = xv.reshape(-1,1))


if __name__ == '__main__':
    gen_data_GRF(1000,Nv = 10,Nu = 30,l=1.0,train = 200, test = 100)