import numpy as np
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

def Euler(F, U, t, dt):
    return U + dt * F(U, t)

def RK4(F, U, t, dt):
    k1 = F(U, t)
    k2 = F(U + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = F(U + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = F(U + dt * k3, t + dt)
    return U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

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

def F_nbody(U, t, masses):

    N = len(masses)
    dim = 2
    state = U.reshape((N, 2*dim))
    positions = state[:, :dim]
    velocities = state[:, dim:]

    accelerations = np.zeros_like(positions)

    for i in range(N):
        acc = np.zeros(dim)
        for j in range(N):
            if i == j:
                continue
            rij = positions[j] - positions[i]
            dist = np.linalg.norm(rij)
            if dist > 1e-12:  # evitar división por cero
                acc += masses[j] * rij / (dist**3)
        accelerations[i] = acc

    # Construir dU/dt
    dUdt = np.empty_like(U)
    for i in range(N):
        dUdt[4*i:4*i+2] = velocities[i]
        dUdt[4*i+2:4*i+4] = accelerations[i]
    return dUdt


# Ejemplo: Sistema Tierra-Luna

# Masas
M_sun = 1.0
M_earth = 3.0e-6   # 1/330000
M_moon = 3.7e-8    # 1/27000000

masses = np.array([M_sun, M_earth, M_moon])

# Condiciones iniciales

# Posiciones y velocidades en el plano XY
r_sun = np.array([0.0, 0.0])
v_sun = np.array([0.0, 0.0])

# Tierra orbitando al Sol a distancia 1, con velocidad orbital = 1
r_earth = np.array([1.0, 0.0])
v_earth = np.array([0.0, 1.0])

# Luna orbitando a la Tierra a distancia 0.0026 (≈ 1/384 de 1 UA), velocidad relativa
r_moon = r_earth + np.array([0.0, 0.0026])
v_moon = v_earth + np.array([-0.04, 0])  # velocidad orbital en -y

# Empaquetar en U0
U0 = np.hstack([r_sun, v_sun, r_earth, v_earth, r_moon, v_moon])

# Integración

t0 = 0.0
tf = np.pi*2  # 2 años
N = 2000  # dt = 0.001

# Envolver F_nbody para que solo dependa de (U, t)
def F(U, t):
    return F_nbody(U, t, masses)

t_vals, U_vals = Cauchy_Solver(F, U0, t0, tf, N, RK4)

# Graficas

plt.figure(figsize=(8, 8))
colors = ['orange', 'blue', 'red']
labels = ['Sol', 'Tierra', 'Luna']

for i in range(3):
    x = U_vals[:, 4*i]
    y = U_vals[:, 4*i + 1]
    plt.plot(x, y, color=colors[i], label=labels[i])
    plt.plot(x[0], y[0], 'o', color=colors[i])  # posición inicial

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title("Simulación del problema de 3 cuerpos")
plt.axis('equal')
plt.grid(True, linestyle='-', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Extraer posiciones
x_sun = U_vals[:, 0]
y_sun = U_vals[:, 1]
x_earth = U_vals[:, 4]
y_earth = U_vals[:, 5]
x_moon = U_vals[:, 8]
y_moon = U_vals[:, 9]

# Posición relativa de la Luna respecto a la Tierra
x_moon_rel = x_moon - x_earth
y_moon_rel = y_moon - y_earth

# Graficar órbita de la Luna respecto a la Tierra
plt.figure(figsize=(8, 8))
plt.plot(x_moon_rel, y_moon_rel, 'r-', label='Luna')
plt.plot(0, 0, 'bo', label='Tierra')  # centro de la órbita relativa
plt.xlabel(r'$x_{\text{rel}}$')
plt.ylabel(r'$y_{\text{rel}}$')
plt.title("Órbita de la Luna respecto a la Tierra")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()