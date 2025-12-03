import numpy as np
import matplotlib.pyplot as plt
from physics import *
from temporal_schemes import *
from Cauchy import *

# Condiciones iniciales
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])
U0 = np.concatenate([r0, v0])
t0 = 0.0
tf = 10.0
N = 1000

# Lista de métodos y sus etiquetas
schemes = [
    (Euler, "Euler explícito"),
    (Crank_Nicolson, "Crank Nicolson"),
    (RK4, "RK4"),
    (Inverse_Euler, "Euler implícito")
]

# Grafica

for metodo, etiqueta in schemes:
    t_vals, U_vals = Cauchy_Solver(F, U0, t0, tf, N, metodo)
    plt.plot(U_vals[:, 0], U_vals[:, 1], label=etiqueta)

plt.axis('equal')
plt.title("Órbitas de Kepler")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()