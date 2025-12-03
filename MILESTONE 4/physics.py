import numpy as np
from numpy import *

def F(U, t):
    r = U[0:2]
    dr = U[2:4]
    norm_r = np.linalg.norm(r)
    return np.concatenate([dr, -r / (norm_r**3)])

def Oscilador(U, t):

    x = U[0]
    xdot = U[1]

    return array((xdot, -x))