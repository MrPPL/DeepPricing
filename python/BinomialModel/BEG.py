# The BEG method
import numpy as np
import itertools
from itertools import product
#numerical exampel article
S = [100,100,100]
vol = 0.2
corr = 0.5
r = 0.1
T = 1
K = 100
n = 3
N = 5 #number of steps
stepSize = T/N

u = np.exp(vol*np.sqrt(stepSize))
d = 1/u


# make lattice
def makeLattice(timesteps, d):
    """Make structure of tree by cartesian products"""
    lattice = {} # dictionary to include timestep and possible states.
    for s in range(1,timesteps+1):
        cartesian = [np.arange(s) for i in range(d)] 
        cartesianProd = list(product(*cartesian))
        lattice[s-1] = cartesianProd
    return lattice

lattice = makeLattice(timesteps=N,d=n)

uVec = np.array([np.exp(vol*np.sqrt(stepSize)), np.exp(vol*np.sqrt(stepSize)), np.exp(vol*np.sqrt(stepSize))])
dVec = np.array([np.exp(-vol*np.sqrt(stepSize)), np.exp(-vol*np.sqrt(stepSize)), np.exp(-vol*np.sqrt(stepSize))])

tree = {}

for key in lattice:
    list1 = []
    for vec in lattice[key]:
        list1.append( S* ((uVec**vec) * dVec**(-(np.array(vec)-key)))  )
    tree[key] = list1

