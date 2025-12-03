import matplotlib.pyplot as plt
from numpy import *
from physics import *
from initial_conditions import *
from temporal_schemes import *
from Cauchy import *
from convergence import *
from stability import *

def plot_N_Body_Problem():

    # Variables
    T = 20
    N = 10000
    t = linspace(0, T, N+1)
    Nb = 3
    Nc = 3

    def F(U, t): 
       return N_Body_Problem(U, t, Nb, Nc)     

    # Initial conditions
    U0 = initial_conditions(Nb, Nc)       

    U = Cauchy_Solver(F, U0, t, RK4)
    Us = reshape(U, (N+1, Nb, Nc, 2)) 

    r = Us[:, :, :, 0]
    rs = reshape(r, (N+1, Nb, Nc)) 
    
    for i in range(Nb):
        plt.plot(rs[:, i, 0], rs[:, i, 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title('Órbitas')
    plt.axis('equal')
    plt.grid()
    plt.show()