from numpy import *
from numpy.linalg import *
from math_tools import *

def stability_region(temporal_scheme, x0=-4, xf=2, y0=-4, yf=4, Np=100):

    x = linspace(x0, xf, Np)
    y = linspace(y0, yf, Np)
    rho = zeros((Np, Np))

    U1 = array([1.0], dtype=complex)    
    t1 = 0
    t2 = 1

    for i in range(Np):
        for j in range(Np):
            w = complex(x[i], y[j])           
            
            def F(U, t): 
                return w * U
            
            r = temporal_scheme(U1, t1, t2, F)           
            
            rho[i,j] = abs(r[0])

    return rho, x, y

def stability_Lagrange(F, U_crit, t):

    def f(U):
        return F(U, t)

    J = Jacobian(f, U_crit)

    return eigvals(J)