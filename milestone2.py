import numpy as np
import matplotlib.pyplot as plt

def F(U, t):
    r = U[0:2]
    dr = U[2:4]
    norm_r = np.linalg.norm(r)
    return np.concatenate([dr, -r / (norm_r**3)])

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

# Condiciones iniciales
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])
U0 = np.concatenate([r0, v0])
t0 = 0.0
tf = 10.0
N = 1000

t_vals, U_vals = Cauchy_Solver(F, U0, t0, tf, N, Euler)

plt.plot(U_vals[:, 0], U_vals[:, 1], label="Euler explícito")

t_vals, U_vals = Cauchy_Solver(F, U0, t0, tf, N, Crank_Nicolson)

plt.plot(U_vals[:, 0], U_vals[:, 1], label="Crank Nicolson")

t_vals, U_vals = Cauchy_Solver(F, U0, t0, tf, N, RK4)

plt.plot(U_vals[:, 0], U_vals[:, 1], label="RK4")

t_vals, U_vals = Cauchy_Solver(F, U0, t0, tf, N, Inverse_Euler)

plt.plot(U_vals[:, 0], U_vals[:, 1], label="Euler implícito")
plt.axis('equal')
plt.title("Órbitas de Kepler")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()