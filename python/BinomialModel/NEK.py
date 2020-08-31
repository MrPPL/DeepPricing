#Binomial tree for multidimensionel by Niklas Ekvall (NEK)
# title: A lattice approach for pricing of multivariate contingent claims 
#packages
import numpy as np
import datetime
import scipy.linalg
import itertools
from itertools import product

## tabel 1
# American put min. option 1 year
spot = 100
strike = 100
r=0.1 # instataneous rate of return
vol = 0.2 #volatilities for assets
T = 1 #maturity 
corr = 0.5 #correlation between assets
steps = 3 #timesteps
d = 2 #underlying assets

#calculations
stepSize = T/steps # length of time step
m = 2**d #number of branches from each node
p = 1/m #equal probability for each branch emanating

#make correlation matrix
def createCorrM(vol, corr, d):
    """Inputs: expect a constant volatility and correlation for every asset and between them"""
    corrMatrix = np.diag([vol**2]*d)
    for i in range(corrMatrix.shape[0]):
        for j in range(corrMatrix.shape[1]):
            if j >i:
                corrMatrix[i,j] = corrMatrix[j,i] = (vol**2)*corr
            else:
                continue
    return corrMatrix
#example
corrMatrix = createCorrM(vol=vol, corr = corr, d=d)

# make lattice
def makeLattice(timesteps, d):
    """Make dictionary for each timestep, where each value corresponding to key is a vector. 
    We assume the spot is the same for each asset"""
    lattice = {} # dictionary to include timestep and possible states.
    for s in range(1,timesteps+1):
        cartesian = [np.arange(s) for i in range(d)] 
        cartesianProd = list(product(*cartesian))
        lattice[s-1] = cartesianProd
    return lattice

lattice = makeLattice(timesteps=steps,d=d)

def createTree(corrMatrix, r, d, vol, steps, T, lattice, spot):
    """Creates multidim tree"""
    L = scipy.linalg.cholesky(corrMatrix) #cholesky decompositon of correlation matrix
    invL = np.linalg.inv(L)
    v = np.matmul(invL,(np.array([r]*d)-np.array([vol**2/2]*d)))

    stepSize = T/steps
    upAsset = np.array([np.sqrt(stepSize)+i*stepSize for i in v])
    downAsset = np.array([-np.sqrt(stepSize)+i*stepSize for i in v])
    tree = {}
    u = {}
    for key in lattice:
        if key == 0:
            tree[key] = np.array((spot,)*d)
            u[key] = np.matmul(invL,np.transpose(np.log(tree[key])))
        else:
            # maybe change to power istead of multiplication
            u[key] = (upAsset*lattice[key] + (downAsset*(-(np.array(lattice[key])-key))))
            y = np.matmul(L,np.transpose(u[key]))
            tree[key] = np.transpose(np.transpose(np.exp(y)) * tree[0])
    return tree

Tree = createTree(corrMatrix=corrMatrix, r=r, d=d, vol=vol, steps=steps, T=T, lattice=lattice, spot=spot)
Tree

def callMaxEuro(S,K):
    return max(max(S)-K,0)

valueTree = {}
for key in reversed(range(steps)):
    if key == steps-1:
        udfaldVec = []
        for udfald in range((steps)**d):
            stockPrice = []
            for stock in range(len(Tree[key])):
                stockPrice.append(Tree[key][stock][udfald])
            udfaldVec.append(callMaxEuro(S=stockPrice,K=strike))
        valueTree[key] = udfaldVec
    else:
        pass
        for poss in range(m):
            pass


Tree
valueTree