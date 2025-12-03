from numpy import *
from numpy.linalg import *
from scipy.linalg import *


def derivative(f, x, r, h=1e-7):

    return (f(x + r*h) - f(x - r*h)) / (2 * h)


def Jacobian(f, x):

    J = zeros((len(x), len(x)))

    for j in range(len(x)):
        r = zeros((len(x)))
        r[j] = 1
        J[:, j] = derivative(f, x, r)

    return J


def Gauss(A, b):

    return solve(A, b)


def Newton(f, x0, tol=1e-10, max_iter=50):
    x = x0.copy()
    for _ in range(max_iter):
        fx = f(x)
        if norm(fx) < tol:
            return x
        A = Jacobian(f, x)
        try:
            dx = solve(A, -fx)
        except LinAlgError:
            raise RuntimeError("El Jacobiano es singular")
        x = x + dx
        if norm(dx) < tol:
            return x
    raise RuntimeError("Newton no converge")