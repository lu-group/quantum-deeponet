import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def solve_Advection( nx, nt, u0, xmax = 1, tmax = 1):

    dt = tmax / (nt - 1)
    dx = xmax / (nx - 1)
    # initialize data structures
    u = np.zeros((nx-1,nt))
    # initial condition 
    u[:,0] = u0[:-1]
    # periodic boundary conditions
    I = np.eye(nx - 1)
    I1 = np.roll(I, 1, axis=0)
    #print(I1)
    I2 = np.roll(I, -1, axis=0)
    #print(I2)
    A = I - I1
    B = I1 + I2 -2*I
    # numerical solution
    for n in range(0, nt - 1):
        u[:, n + 1] = u[:, n] - ( dt /   dx )* np.dot(A, u[:, n]) 
    # to make sure u hase periodic boundary condition
    u = np.concatenate([u, u[0:1, :]], axis=0)
    return u 


def gen_train(n_branch=20, n_trunk = 100, N_train = 1000,T = 1, nx = 201, nt = 20001,  length_scale = 1.5):
    # T is the periodicity, n is the number of sensors we picked out
    space = dde.data.GRF(T=T, kernel="ExpSineSquared", length_scale=length_scale, N=nx, interp="cubic") 
    features = space.random(N_train) # features.shape = [N_train, nx]
    sensors = np.linspace(0, 1, num=nx)[:, None] # sensor.shape = [m,1]
    sensor_values = space.eval_batch(features, sensors) # sensor_values.shape = [N_train,m]

    index_x_branch = [round(nx/n_branch)*i for i in range(n_branch-1)]+[nx-1]
    index_x = [round(nx/n_trunk)*i for i in range(n_trunk-1)]+[nx-1]
    index_t = [round(nt/n_trunk)*i for i in range(n_trunk-1)]+[nt-1]
    
    X_branch= sensor_values[:, index_x_branch]
    X_trunk = np.array([[a, b] for a in np.linspace(0, 1, n_trunk) for b in np.linspace(0, 1, n_trunk)])

    def eval_s(u0):
        return solve_Advection( nx, nt, u0, xmax = 1, tmax = 1)

    s = np.array(list(map(eval_s, sensor_values, )))

    y = s[:, index_x][:,:,index_t].reshape(N_train, n_trunk * n_trunk)

    np.savez(f"l={length_scale}_n={n_branch,n_trunk}_data/Advection_train_full.npz", X_train=sensor_values, y_train=s,)
    np.savez(f"l={length_scale}_n={n_branch,n_trunk}_data/Advection_train.npz", X_train0=X_branch, X_train1=X_trunk, y_train=y,)

def gen_test(n_branch=20, n_trunk = 100, N_test = 200,T = 1, nx = 201, nt = 20001,  length_scale = 1.5):
    # T is the periodicity, n is the number of sensors we picked out
    space = dde.data.GRF(T=T, kernel="ExpSineSquared", length_scale=length_scale, N=nx, interp="cubic") 
    features = space.random(N_test) # features.shape = [N_test, nx]
    sensors = np.linspace(0, 1, num=nx)[:, None] # sensor.shape = [m,1]
    sensor_values = space.eval_batch(features, sensors) # sensor_values.shape = [N_test,m]

    index_x_branch = [round(nx/n_branch)*i for i in range(n_branch-1)]+[nx-1]
    index_x = [round(nx/n_trunk)*i for i in range(n_trunk-1)]+[nx-1]
    index_t = [round(nt/n_trunk)*i for i in range(n_trunk-1)]+[nt-1]
    
    X_branch= sensor_values[:, index_x_branch]
    X_trunk = np.array([[a, b] for a in np.linspace(0, 1, n_trunk) for b in np.linspace(0, 1, n_trunk)])

    def eval_s(u0):
        return solve_Advection( nx, nt, u0, xmax = 1, tmax = 1)

    s = np.array(list(map(eval_s, sensor_values, )))

    y = s[:, index_x][:,:,index_t].reshape(N_test, n_trunk * n_trunk)

    np.savez(f"l={length_scale}_n={n_branch,n_trunk}_data/Advection_test_full.npz", X_test=sensor_values, y_test=s,)
    np.savez(f"l={length_scale}_n={n_branch,n_trunk}_data/Advection_test.npz", X_test0=X_branch, X_test1=X_trunk, y_test=y,)

if __name__ == '__main__':
    gen_train(n_branch=20, n_trunk = 50,length_scale=1.5)
    gen_test(n_branch=20, n_trunk = 50,length_scale=1.5)
