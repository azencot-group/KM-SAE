import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import t_to_np, imshow_seqeunce
import numpy as np


def get_unique_num(D, I, static_number):
    """ This function gets a parameter for number of unique components. Unique is a component with imag part of 0 or
        couple of conjugate couple """
    i = 0
    for j in range(static_number):
        index = len(I) - i - 1
        val = D[I[index]]

        if val.imag == 0:
            i = i + 1
        else:
            i = i + 2

    return i


def get_sorted_indices(D, pick_type):
    """ Return the indexes of the eigenvalues (D) sorted by the metric chosen by an hyperparameter"""

    if pick_type == 'real':
        I = np.argsort(np.real(D))
    elif pick_type == 'norm':
        I = np.argsort(np.abs(D))
    elif pick_type == 'ball' or pick_type == 'space_ball':
        Dr = np.real(D)
        Db = np.sqrt((Dr - np.ones(len(Dr))) ** 2 + np.imag(D) ** 2)
        I = np.argsort(Db)
    else:
        raise Exception("no such method")

    return I


def static_dynamic_split(D, I, pick_type, static_size):
    """Return the eigenvalues indexes of the static and dynamic factors"""
    if pick_type == 'ball' or pick_type == 'space_ball':
        static_size = get_unique_num(D, I[::-1], static_size)
        Is, Id = I[:static_size], I[static_size:]
    else:
        static_size = get_unique_num(D, I, static_size)
        Id, Is = I[:-static_size], I[-static_size:]
    return Id, Is


def swap(model, X, Z, C, indices, static_size, plot=False, pick_type='norm'):
    """Swaps between two samples in a batch by the indices given
        :param model - the trained model to use in the swap
        :param X - the original samples, used for displaying the original
        :param Z - the latent representation
        :param C - The koopman matrix. Used to project into the subspaces
        :param indices - indexes for choosing a pair from the batch
        :param static_size - the number of eigenvalues that are dedicated to the static subspace.
         The rest will be for the dynamic subspace
        :param plot - plot with matplotlib
        :param pick_type - the metric to pick the static eigenvalues"""

    # swap a single pair in batch
    bsz, fsz = X.shape[0:2]
    device = X.device

    # swap contents of samples in indices
    X = t_to_np(X)
    Z = t_to_np(Z.reshape(bsz, fsz, -1))
    C = t_to_np(C)

    ii1, ii2 = indices[0], indices[1]

    S1, Z1 = X[ii1].squeeze(), Z[ii1].squeeze()
    S2, Z2 = X[ii2].squeeze(), Z[ii2].squeeze()

    # eig
    D, V = np.linalg.eig(C)
    U = np.linalg.inv(V)

    # project onto V
    Zp1, Zp2 = Z1 @ V, Z2 @ V

    # static/dynamic split
    I = get_sorted_indices(D, pick_type)
    Id, Is = static_dynamic_split(D, I, pick_type, static_size)

    # Zp* is in t x k
    Z1d, Z1s = Zp1[:, Id] @ U[Id], Zp1[:, Is] @ U[Is]
    Z2d, Z2s = Zp2[:, Id] @ U[Id], Zp2[:, Is] @ U[Is]

    Z1d2s = np.real(Z1d + Z2s)
    Z2d1s = np.real(Z2d + Z1s)

    # reconstruct
    S1d2s = model.decode(torch.from_numpy(Z1d2s.reshape((fsz, -1, 1, 1))).to(device))
    S2d1s = model.decode(torch.from_numpy(Z2d1s.reshape((fsz, -1, 1, 1))).to(device))

    # visualize
    if plot:
        titles = ['S{}'.format(ii1), 'S{}'.format(ii2), 'S{}d{}s'.format(ii2, ii2), 'S{}d{}s'.format(ii2, ii1)]
        imshow_seqeunce([[S1], [S2], [S1d2s.squeeze()], [S2d1s.squeeze()]],
                        plot=plot, titles=np.asarray([titles]).T, figsize=(50, 10), fontsize=50)


def swap_by_index(model, X, Z, C, indices, Sev_idx, Dev_idx, plot=False):
    """ Transfer specific features using static eigenvectors indices and dynamic eigenvectors indices
        Can be used for example to illustrate the multi-factor disentanglement
        indices - tuple of 2 samples
        Sev_idx - static eigenvectors indices
        Dev_idx - dynamic eigenvectors indices
        X - batch of samples
        Z - latent features of the batch """
    # swap a single pair in batch
    bsz, fsz = X.shape[0:2]
    device = X.device

    # swap contents of samples in indices
    X = t_to_np(X)
    Z = t_to_np(Z.reshape(bsz, fsz, -1))
    C = t_to_np(C)

    ii1, ii2 = indices[0], indices[1]
    S1, Z1 = X[ii1].squeeze(), Z[ii1].squeeze()
    S2, Z2 = X[ii2].squeeze(), Z[ii2].squeeze()

    # eig
    D, V = np.linalg.eig(C)
    U = np.linalg.inv(V)

    # project onto V
    Zp1, Zp2 = Z1 @ V, Z2 @ V

    # static/dynamic split
    Id, Is = Dev_idx, Sev_idx

    # Zp* is in t x k
    Z1d, Z1s = Zp1[:, Id] @ U[Id], Zp1[:, Is] @ U[Is]
    Z2d, Z2s = Zp2[:, Id] @ U[Id], Zp2[:, Is] @ U[Is]

    # swap
    Z1d2s = np.real(Z1d + Z2s)
    Z2d1s = np.real(Z2d + Z1s)

    # reconstruct
    S1d2s = model.decode(torch.from_numpy(Z1d2s.reshape((fsz, -1, 1, 1))).to(device))
    S2d1s = model.decode(torch.from_numpy(Z2d1s.reshape((fsz, -1, 1, 1))).to(device))

    # visualize
    if plot:
        titles = ['S{}'.format(ii1), 'S{}'.format(ii2), 'S{}d{}s'.format(ii2, ii2), 'S{}d{}s'.format(ii2, ii1),
                  'S{}s'.format(ii1), 'S{}s'.format(ii2), 'S{}d'.format(ii1), 'S{}d'.format(ii2)]
        imshow_seqeunce([[S1], [S2], [S1d2s.squeeze()], [S2d1s.squeeze()]],
                        plot=plot, titles=np.asarray([titles[:4]]).T, figsize=(50, 10), fontsize=50)

    return S1d2s, S2d1s, Z1d2s, Z2d1s
