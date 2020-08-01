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

