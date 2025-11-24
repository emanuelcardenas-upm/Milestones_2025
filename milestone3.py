import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import root
import matplotlib.pyplot as plt

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

def F(U, t):
    r = U[0:2]
    dr = U[2:4]
    norm_r = np.linalg.norm(r)
    if norm_r < 1e-12:
        # Evita división por cero
        a = np.array([0.0, 0.0])
    else:
        a = -r / (norm_r**3)
    return np.concatenate([dr, a])

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

# Condiciones iniciales
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])
U0 = np.concatenate([r0, v0])
t0 = 0.0
tf = 10.0

esquemas = {
    "Euler": Euler,
    "Inverse_Euler": Inverse_Euler,
    "Crank_Nicolson": Crank_Nicolson,
    "RK4": RK4
}

N_vals = [125, 250, 500, 1000, 2000, 5000, 10000]

errores = {nombre: [] for nombre in esquemas}
pasos = {nombre: [] for nombre in esquemas}

print("Calculando errores con Richardson...")
for nombre, metodo in esquemas.items():
    print(f"Procesando {nombre}...")
    for N in N_vals:
        try:
            _, U_N = Cauchy_Solver(F, U0, t0, tf, N, metodo)
            _, U_2N = Cauchy_Solver(F, U0, t0, tf, 2*N, metodo)

            error = np.linalg.norm(U_2N[-1] - U_N[-1])
            errores[nombre].append(error)
            pasos[nombre].append((tf - t0) / N)
        except Exception as e:
            print(f"  Error con {nombre} y N={N}: {e}")
            errores[nombre].append(np.nan)
            pasos[nombre].append((tf - t0) / N)

plt.figure(figsize=(10, 6))

for nombre in esquemas:
    dt_vals = np.array(pasos[nombre])
    err_vals = np.array(errores[nombre])
    
    validos = ~np.isnan(err_vals)
    dt_vals = dt_vals[validos]
    err_vals = err_vals[validos]

    # Calcular tasa de convergencia (pendiente en log-log)
    if len(err_vals) > 1:
        p = np.polyfit(np.log(dt_vals), np.log(err_vals), 1)[0]
        plt.loglog(dt_vals, err_vals, 'o-', label=f"{nombre} (p={p:.2f})")
    else:
        plt.loglog([], [], label=f"{nombre} (no converge)")

# Líneas de referencia para órdenes 1, 2, 4
dt_ref = np.array([1e-3, 1e-1])
plt.loglog(dt_ref, dt_ref**1, 'm-.', label="Orden 1")
plt.loglog(dt_ref, dt_ref**2, 'k-.', label="Orden 2")
plt.loglog(dt_ref, dt_ref**4, '-.', label="Orden 4")

plt.xlabel("dt")
plt.ylabel("Error estimado " r'$(||U_{2N} - U_N||)$')
plt.title("Tasa de convergencia")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.show()