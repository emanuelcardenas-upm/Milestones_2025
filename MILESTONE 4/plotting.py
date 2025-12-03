import matplotlib.pyplot as plt
from numpy import *
from physics import *
from temporal_schemes import *
from Cauchy import *
from convergence import *
from stability import *

def plot_Cauchy_step_1():

    # Variables
    T = 20
    N = 10000
    t = linspace(0, T, N+1)

    # Condiciones iniciales
    U0 = ([0, 1])        

    U = Cauchy_Solver(Oscilador, U0, t, LeapFrog_step_1(U0, Oscilador))
    
    plt.plot(U[:, 0], U[:, 1]) 
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title('Órbita (Cauchy 1 step)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_Cauchy_step_2():

    # Variables
    T = 20
    N = 10000
    t = linspace(0, T, N+1)

    # Condiciones iniciales
    U0 = ([0, 1]) 
    U1 = Euler(U0, t[0], t[1], Oscilador)           

    U = Cauchy_solver_2_steps(Oscilador, U0, U1, t, LeapFrog_step_2)
    
    plt.plot(U[:, 0], U[:, 1])  
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title('Órbita (Cauchy 2 step)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_Stability():
    
    rho, x, y = stability_region(RK4)    
    
    plt.contour(x, y, transpose(rho), linspace(0, 1, 11))
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.title('Estabilidad')
    plt.axis('equal')
    plt.grid()
    plt.show()