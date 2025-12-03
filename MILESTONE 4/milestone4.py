import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Configuración estilo
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "legend.fontsize": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

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
    def G(U_new):
        return U_new - U - dt * F(U_new, t + dt)
    U_guess = U + dt * F(U, t)
    sol = root(G, U_guess, method='hybr', tol=tol)
    if not sol.success:
        sol = root(G, U_guess, method='lm')
    if not sol.success:
        raise RuntimeError(f"Inverse_Euler falló: {sol.message}")
    return sol.x

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

def LeapFrog_step(F, U, t, dt, acc_prev=None):
    x, v = U
    if acc_prev is None:
        acc = F(U, t)[1]
    else:
        acc = acc_prev
    v_half = v + 0.5 * dt * acc
    x_new = x + dt * v_half
    acc_new = F([x_new, v_half], t + dt)[1]
    v_new = v_half + 0.5 * dt * acc_new
    return np.array([x_new, v_new]), acc_new

def integrate_leapfrog(F, U0, t0, tf, N):
    dt = (tf - t0) / N
    t = t0
    U = np.array(U0, dtype=float)
    t_vals = [t]
    U_vals = [U.copy()]
    acc = F(U, t)[1]
    for _ in range(N):
        U, acc = LeapFrog_step(F, U, t, dt, acc_prev=acc)
        t += dt
        t_vals.append(t)
        U_vals.append(U.copy())
    return np.array(t_vals), np.array(U_vals)

def F_osc(U, t):
    x, v = U
    return np.array([v, -x])

U0 = [1.0, 0.0]
t0 = 0.0
tf = 20.0
N = 200

schemes = {
    "Euler": Euler,
    "Inverse_Euler": Inverse_Euler,
    "Crank_Nicolson": Crank_Nicolson,
    "RK4": RK4
}

# Gráficas
plt.figure(figsize=(12, 10))
plot_idx = 1

for name, scheme in schemes.items():
    t_vals, U_vals = Cauchy_Solver(F_osc, U0, t0, tf, N, scheme)
    plt.subplot(3, 2, plot_idx)
    plt.plot(t_vals, U_vals[:, 0], label=f"{name}", color="blue")
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x(t)$')
    plt.grid(True)
    plt.legend()
    plot_idx += 1

t_vals, U_vals = integrate_leapfrog(F_osc, U0, t0, tf, N)
plt.subplot(3, 2, plot_idx)
plt.plot(t_vals, U_vals[:, 0], label="Leap-Frog", color="blue")
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()

# Estabilidad

def R_euler(z):       return 1 + z
def R_inv_euler(z):   return 1 / (1 - z)
def R_cn(z):          return (1 + z/2) / (1 - z/2)
def R_rk4(z):         return 1 + z + z**2/2 + z**3/6 + z**4/24

def modulus_R(method, dt):
    z = 1j * dt
    if method == "Euler":
        return np.abs(1 + z)
    if method == "Inverse_Euler":
        return np.abs(1 / (1 - z))
    if method == "Crank_Nicolson":
        return np.abs((1 + z/2) / (1 - z/2))
    if method == "RK4":
        return np.abs(1 + z + z**2/2 + z**3/6 + z**4/24)
    if method == "Leap-Frog":
        if dt <= 2:
            return 1.0
        else:
            a = (2 - dt**2) / 2
            disc = a**2 - 1
            rho1 = a + np.sqrt(disc)
            return abs(rho1)

dts = np.linspace(0, 3, 400)
plt.figure(figsize=(10, 6))
for name in list(schemes.keys()) + ["Leap-Frog"]:
    mags = [modulus_R(name, dt_val) for dt_val in dts]
    plt.plot(dts, mags, label=name)

plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$|R(\lambda \Delta t)|$')
plt.title("Módulo del factor de amplificación (estabilidad absoluta)")
plt.grid(True)
plt.legend()
plt.ylim(0, 2)
plt.show()