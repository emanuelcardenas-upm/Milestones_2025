import numpy as np

def Euler(F, U, t, dt):
    return U + dt * F(U, t)

def Crank_Nicolson(F, U, t, dt, tol=1e-12, max_iter=50):
    U_new = U.copy()
    F_n = F(U, t)
    for _ in range(max_iter):
        F_new = F(U_new, t + dt)
        U_next = U + (dt / 2.0) * (F_n + F_new)
        if np.linalg.norm(U_next - U_new) < tol:
            return U_next
        U_new = U_next
    raise RuntimeError(f"Crank-Nicolson no convergió en {max_iter} iteraciones.")

def RK4(F, U, t, dt):
    k1 = F(U, t)
    k2 = F(U + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = F(U + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = F(U + dt * k3, t + dt)
    return U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def Inverse_Euler(F, U, t, dt, tol=1e-12, max_iter=50):
    U_new = U.copy()
    for _ in range(max_iter):
        U_next = U + dt * F(U_new, t + dt)
        if np.linalg.norm(U_next - U_new) < tol:
            return U_next
        U_new = U_next
    raise RuntimeError(f"Euler implícito no convergió en {max_iter} iteraciones.")