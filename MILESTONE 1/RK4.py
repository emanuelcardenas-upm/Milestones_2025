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

# Definicion de variables

U[0:4] = [*r0, *dr0]

x_vals = [U[0]]
y_vals = [U[1]]

# Codigo RK4

for i in range(N):
    k1 = F(U)
    k2 = F(U + (dt/2) * k1)
    k3 = F(U + (dt/2) * k2)
    k4 = F(U + dt * k3)
    
    U_new = U + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    U = U_new
        
    x_vals.append(U[0])
    y_vals.append(U[1])

# Grafica

plt.plot(x_vals, y_vals, label = "RK4")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbitas de Kepler")
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()
