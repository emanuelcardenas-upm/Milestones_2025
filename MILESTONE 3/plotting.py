import matplotlib.pyplot as plt
from numpy import *
from F import *
from temporal_schemes import *
from Cauchy import *
from convergence import *

def plot_error():

    # Variables
    U0 = ([1, 0, 0, 1])
    T = 20
    N = 10000
    t = linspace(0, T, N+1)
    q = 1

    U, E = Cauchy_error(F, U0, t, RK4, q)

    plt.plot(U[:, 0], U[:, 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title('Órbita')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    plt.plot(t, E[:, 0])
    plt.xlabel(r'$t$')
    plt.ylabel('Error')
    plt.title('Oscilador')    
    plt.grid(True)
    plt.show()


def plot_convergence():    

    U0 = ([1, 0])
    T = 8*pi
    N = 1000
    t = linspace(0, T, N+1)

    logN, logE, q = convergence_rate(RK4, Oscilador, U0, t)   

    plt.plot(logN, logE)
    plt.xlabel('logN')
    plt.ylabel('logE')
    plt.title(f'El orden del esquema temporal es {q}')
    plt.grid(True)
    plt.show()