import matplotlib.pyplot as plt
from numpy import *
from physics import *
from initial_conditions import *
from temporal_schemes import *
from Cauchy import *
from convergence import *
from stability import *

def plot_CR3BP():

    def F(U, t):
        return CR3BP(U, t, mu)

    T = 100
    N = 10000
    t = linspace(0, T, N+1)
    mu = 0.0121505856  # Sistema Tierra-Luna por ejemplo
    U0 = array([0.8, 0, 0, 0.5, 0.3, 0])

    U = Cauchy_Solver(F, U0, t, RK56)

    # Primarios
    m1 = array([-mu, 0])
    m2 = array([1 - mu, 0])

    plt.figure(figsize=(10,5))
    plt.plot(m1[0], m1[1], 'o', color='gray', label=r'M$_1$ (primario)')    
    plt.plot(m2[0], m2[1], 'o', color='blue', label=r'M$_2$ (secundario)')
    
    plt.plot(U[:, 0], U[:, 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title('Órbitas')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_Lagrange_Points():

    mu = 0.0121505856  # Sistema Tierra-Luna por ejemplo

    L1, L2, L3, L4, L5 = Lagrange_Points(mu)

    # Primarios
    m1 = array([-mu, 0])
    m2 = array([1 - mu, 0])

    plt.figure(figsize=(10,5))
    plt.axhline(0, color='gray', linewidth=0.5)
    
    plt.plot(m1[0], m1[1], 'o', color='black', label=r'M$_1$ (primario)')    
    plt.plot(m2[0], m2[1], 'o', color='gray', label=r'M$_2$ (secundario)')

    # Puntos de Lagrange
    plt.plot(L1[0], L1[1], 'ro', label=r"L$_1$")
    plt.plot(L2[0], L2[1], 'ro', label=r"L$_2$")
    plt.plot(L3[0], L3[1], 'ro', label=r"L$_3$")
    plt.plot(L4[0], L4[1], 'bo', label=r"L$_4$")
    plt.plot(L5[0], L5[1], 'bo', label=r"L$_5$")
    
    plt.title("Puntos de Lagrange en el CR3BP")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def print_stability():

    def F(U, t):
        return CR3BP(U, t, mu)
    
    T = 100
    N = 10000
    t = linspace(0, T, N+1)
    mu = 0.0121505856  # Sistema Tierra-Luna

    L1, L2, L3, L4, L5 = Lagrange_Points(mu)

    eigenvalues = zeros((5, 6), dtype=complex) # 6 autovalores por cada punto de Lagrange
    for i, L in enumerate([L1, L2, L3, L4, L5]):
        U_crit = array([L[0], L[1], L[2], 0, 0, 0]) # Punto de equilibrio
        eigenvalues[i, :] = stability_Lagrange(F, U_crit, t)   

        print(f"Autovalores en L{i+1}:")
        for j in range(6):            
            print(f"λ_{j+1}: ", eigenvalues[i, j])