from numpy import *
from numpy.linalg import *
from Cauchy import *
from temporal_schemes import *
from convergence import *
from config_plotting import *
from plotting import *

config_plotting()

plot_error()
plot_convergence()