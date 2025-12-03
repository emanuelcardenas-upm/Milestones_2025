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

tol = 1e-15
max_iter = 50

x_vals = [U[0]]
y_vals = [U[1]]

# Codigo C-N

for i in range(N):
    Fn = F(U)
    U_new = U.copy()
    
    for j in range(max_iter):
        F_new = F(U_new)
        U_next = U + (dt/2) * (Fn + F_new)

        error = np.linalg.norm(U_next - U_new)
        
        U_new = U_next
        
        if error < tol:
            break
    else:
        print("Paso", i+1 , ": no convergió en", max_iter, "iteraciones")
    
    U = U_new
    
    x_vals.append(U[0])
    y_vals.append(U[1])

# Grafica

plt.plot(x_vals, y_vals, label = "Crank-Nicolson")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbitas de Kepler")
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()
