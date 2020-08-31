import math
import numpy as np
#from scipy.interpolate import interp1d
from copy import deepcopy


def Currin(x, d):
    return -1*float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20))))

def branin(x1,d):
    x=deepcopy(x1)
    x[0]= 15* x[0]-5
    x[1]=15*x[1]
    return -1*float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10)