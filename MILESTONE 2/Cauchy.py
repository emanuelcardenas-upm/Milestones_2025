import numpy as np

def Cauchy_Solver(F, U0, t0, tf, N, scheme):
    if N <= 0:
        raise ValueError("N debe ser un entero positivo.")
    
    dt = (tf - t0) / N
    t_vals = [t0]
    U_vals = [np.array(U0, dtype=float)]
    U = np.array(U0, dtype=float)
    t = t0

    for _ in range(N):
        U = scheme(F, U, t, dt)
        t += dt
        t_vals.append(t)
        U_vals.append(U.copy())

    return np.array(t_vals), np.array(U_vals)