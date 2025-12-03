from numpy import array, fabs
import numpy as np
import matplotlib.pyplot as plt

# Funcion F(U)

def F(U):
    r = U[0:2]
    dr = U[2:4]
    return np.array([*dr, *-r/((np.linalg.norm(r))**3)])

# Definicion del paso de tiempo

T = 10
N = 1000
dt = T/N

# Definicion de variables

U = np.zeros(4)

# Condiciones iniciales

r0 = np.array([1, 0])
dr0 = np.array([0, 1])

################################################################################
############################# Euler Explicito ##################################
################################################################################

# Definicion de variables

U[0:4] = [*r0, *dr0]

x_vals = [U[0]]
y_vals = [U[1]]

# Codigo EE

for i in range(N):
    U_new = U + dt * F(U)
    
    U = U_new
    
    x_vals.append(U[0])
    y_vals.append(U[1])

# Grafica

plt.plot(x_vals, y_vals, label = "Euler Explicito")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbitas de Kepler")
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.show()
