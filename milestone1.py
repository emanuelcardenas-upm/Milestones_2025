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

################################################################################
############################## Crank-Nicolson ##################################
################################################################################

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

################################################################################
##################### Runge-Kutta de cuarto orden - RK4 ########################
################################################################################

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

plt.plot(x_vals, y_vals, linestyle=':', label = "RK4")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbitas de Kepler")
plt.axis('equal')
plt.legend(loc = 'upper right')
plt.show()
