"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           Monte Carlo simulation of path of stock with constant volatility and risk free rate
// Course:          Master thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
//
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Valuing American Options by Simulation: A Simple Least-Squares Approach by Longstaff and Schwartz
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
"""
import numpy as np
import pandas as pd



# Variables in American put example in Longstaff
#r= 0.06
#vol = 0.2
#spot = 36
#strike = 40
#T = 1 # maturity in years
#timePointsYear = 50
#n = 100 # simulation of paths
#function for stock path over timepoints
def stockStep(S, r, vol, timeStep, normRV): 
    exponent = (r-vol**2/2)*timeStep+vol* normRV*np.sqrt(timeStep)
    return S*np.exp(exponent)

#Simulation
def simStockPath(spot, r, vol, timePointsYear, T, n):  
    """This function simulate pathwise stock prices, given spot, rate (r), volatility (vol),
     timePoints, T (maturity), n = number of paths (all types floats)"""
    timePoints = timePointsYear * T # total number of timepoints
    stockPaths = np.zeros((n, timePoints+1))
    timeStep = 1/timePointsYear
    for path in stockPaths:
        normRV = np.random.normal(loc=0, scale=1, size=len(path)-1)
        for j in range(len(path)):
            if j == 0:
                path[j] = spot
            else:
                path[j] = stockStep(S, r, vol, timeStep, normRV[j-1])
            S = path[j] #updating stock price
    return (stockPaths)


# Simulate pathStock
stockMatrix = simStockPath(spot=36, r=0.06, vol=0.2, timePointsYear=5, T=1, n=5)


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import simPathStock

#Variables for american put
spot=36
r=0.06
vol=0.2
timePointsYear=1000
T=1
n= 10**4
strike = 40  
steps = timePointsYear*T

#stockMatrix
import numpy.random as npr
xi = npr.normal(0,np.sqrt(1/steps),(n,steps)) #simulate normal fordelt with T/steps standard deviation
W = np.apply_along_axis(np.cumsum,1,xi) # sum r.v. normals row wise
W = np.concatenate((np.zeros((n,1)),W),1) # add zero as initial value
drift = np.linspace(0,r-(vol**2)/2,steps+1) # drift of GBM: r-(1/2) * vol**2
drift = np.reshape(drift,(1,steps+1)) # add zero 
drift = np.repeat(drift,n,axis=0) # make same dimensions
S = spot * np.exp(drift + vol * W)


dS = np.diff(S,1,1)
tim = np.linspace(0,1,steps+1)

tSinput = []
for i in range(steps):
    timv = np.repeat(tim[i],n)
    timv = np.reshape(timv,(n,1))
    Sv = np.reshape(S[:,i],(n,1))
    tSinput.append(np.concatenate((timv,Sv),1))

###############
### Illustrate simulated paths
##############
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize']=5,4
plt.style.use('ggplot')
plt.plot(tim,S[0],tim,S[1],tim,S[2])
plt.xlabel("Time t")
plt.ylabel("Stock value")
plt.title("Sample paths for stock")
plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/latex/Figures/samplePath.pdf")
plt.show()

