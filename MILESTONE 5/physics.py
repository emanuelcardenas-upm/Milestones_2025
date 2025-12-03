import numpy as np
from numpy import *
from numpy.linalg import norm

def F(U, t):
    r = U[0:2]
    dr = U[2:4]
    norm_r = np.linalg.norm(r)
    return np.concatenate([dr, -r / (norm_r**3)])

def Oscilador(U, t):

    x = U[0]
    xdot = U[1]

    return array((xdot, -x))

def N_Body_Problem(U, t, Nb, Nc):

    Us = reshape(U, (Nb, Nc, 2))
    r = reshape(Us[:,:,0], (Nb, Nc))
    v = reshape(Us[:,:,1], (Nb, Nc))

    F = zeros(Nb*Nc*2)
    Fs = reshape(F, (Nb, Nc, 2))
    drdt = reshape(Fs[:,:,0], (Nb, Nc))
    dvdt = reshape(Fs[:,:,1], (Nb, Nc))

    for i in range(Nb):
        drdt[i,:] = v[i,:]

        for j in range(Nb):
            if i != j:
                dvdt[i,:] += (r[j,:] - r[i,:]) / norm(r[j,:] - r[i,:])

    return F