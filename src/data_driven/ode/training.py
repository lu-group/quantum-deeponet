import numpy as np
import deepxde as dde
from ...classical_orthogonal_deeponet import OrthoONetCartesianProd


def trunk_transform(x, trunk_min, trunk_max):
    d = x.shape[1]
    x = 2 * (x - trunk_min) / (trunk_max - trunk_min) - 1  # Rescale to [-1, 1]
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)

def branch_transform(x, branch_min, branch_max):
    # For 2-dimensional input
    d = x.shape[1]
    x = 2 * (x - branch_min) / (branch_max - branch_min) - 1  # Rescale to [-1, 1]
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)

def main():
    d1 = np.load(r'picked_aligned_train.npz',allow_pickle=True)
    x_train,y_train = (d1['X0'].astype(np.float32),d1['X1'].astype(np.float32)),d1['y'].astype(np.float32)
    d2 = np.load(r'picked_aligned_test.npz',allow_pickle=True)
    x_test,y_test = (d2['X0'].astype(np.float32),d2['X1'].astype(np.float32)),d2['y'].astype(np.float32)

    trunk_min = np.min( np.stack((np.min(x_train[1], axis=0),np.min(x_test[1], axis=0)),axis=0),axis=0)
    trunk_max = np.max( np.stack((np.max(x_train[1], axis=0),np.max(x_test[1], axis=0)),axis=0),axis=0)
    branch_min = np.min(np.stack((np.min(x_train[0], axis=0),np.min(x_test[0], axis=0)),axis=0),axis=0)
    branch_max = np.max( np.stack((np.max(x_train[0], axis=0),np.max(x_test[0], axis=0)),axis=0),axis=0)

    x_train = (branch_transform(x_train[0], branch_min, branch_max),
               trunk_transform(x_train[1]), trunk_min, trunk_max)
    x_test = (branch_transform(x_test[0]), branch_min, branch_max,
              trunk_transform(x_test[1]), trunk_min, trunk_max)

    # import pdb;pdb.set_trace()
    data = dde.data.TripleCartesianProd(
        X_train = x_train, y_train = y_train, X_test = x_test, y_test = y_test
    )
    #choose network
    m = 10
    dim_x = 1
    net = OrthoONetCartesianProd(
        [m+1,10,10],
        [dim_x+1,10,10],
        'relu'
    )

    model = dde.Model(data,net)
    model.compile('adam',lr=0.001,metrics=['mean l2 relative error'])
    losshistory, train_state = model.train(iterations=30000,disregard_previous_best = True)


    dde.utils.external.save_loss_history(losshistory,r'classical_training/loss_history.txt' )
    dde_model = model.net
    model.save(r'classical_training/model_checkpoint')
    for name,param in dde_model.named_parameters():
        np.savetxt(fr'{name}.txt',param.cpu().detach().numpy())


if __name__ == '__main__':
    main()