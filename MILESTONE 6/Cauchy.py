import numpy as np
from numpy import *

def Cauchy_Solver(F, U0, t, temporal_scheme):

    N = len(t) - 1
    Nv = len(U0)

    U = zeros((N+1, Nv))
    U[0,:] = U0

    for n in range(N):
        U[n+1,:] = temporal_scheme(U[n,:], t[n], t[n+1], F)
    return U

def Cauchy_solver_2_steps(F, U0, U1, t, temporal_scheme):

    N = len(t) - 1
    Nv = len(U0)

    U = zeros((N+1, Nv))
    U[0,:] = U0
    U[1,:] = U1

    for n in range(1, N):
        U[n+1,:] = temporal_scheme(U[n-1,:], U[n,:], t[n], t[n+1], F)
    return U

def Cauchy_error(F, U0, t, temporal_scheme, q=1):

    N = len(t) - 1
    Nv = len(U0)

    E = zeros((N+1, Nv))
    t1 = t[:]
    t2 = linspace(t[0], t[N], 2*N+1)

    U1 = Cauchy_Solver(F, U0, t1, temporal_scheme)
    U2 = Cauchy_Solver(F, U0, t2, temporal_scheme)

    for n in range(0, N+1):
        E[n,:] = (U2[2*n,:] - U1[n,:])/(1-1/2**q)
    return U1, E