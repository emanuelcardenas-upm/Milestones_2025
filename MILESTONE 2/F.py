import numpy as np

def F(U, t):
    r = U[0:2]
    dr = U[2:4]
    norm_r = np.linalg.norm(r)
    return np.concatenate([dr, -r / (norm_r**3)])