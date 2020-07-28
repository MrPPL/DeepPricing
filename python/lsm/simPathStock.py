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
r= 0.06
vol = 0.2
spot = 36
strike = 40
T = 1 # maturity in years
timePointsYear = 50
n = 100 # simulation of paths
#function for stock path over timepoints
stockStep = lambda S, r, vol, timeStep, normRV: S*np.exp((r-np.power(vol,2)/2)*timeStep+vol* normRV*np.sqrt(timeStep))

#Simulation
def simStockPath(spot, r, vol, timePointsYear, T, n):  
    """This function simulate pathwise stock prices, given spot, rate (r), volatility (vol),
     timePoints, T (maturity), n = number of paths (all types floats)"""
    timePoints = timePointsYear * T # total number of timepoints
    stockPaths = np.zeros((n, timePoints+1))
    timeStep = 1/timePointsYear
    for path in stockPaths:
        normRV = np.random.normal(loc=0, scale=1, size=len(path))
        for j in range(len(path)):
            if j == 0:
                path[j] = spot
            else:
                path[j] = stockStep(S, r, vol, timeStep, normRV[j])
            S = path[j] #updating stock price
    return (stockPaths)



from sklearn.linear_model import LinearRegression

#Variables for american put
spot=36
r=0.06
vol=0.2
timePointsYear=5
strike = 40
T=1
n=5**1

# Simulate pathStock
stockMatrix = simStockPath(spot, r, vol, timePointsYear, T, n)


#contract function
putCall = lambda S, K: K-S if (K > S) else 0 

#Intrinsic matrix
def intrinsic(spot, r, vol, timePointsYear, strike, T, n):
    """ Finds the intrinsic value matrix """
    stockMatrix = simStockPath(spot, r, vol, timePointsYear, T, n)
    timePoints = T * timePointsYear
    intrinsicM = np.zeros((n, timePoints+1))
    count=1
    for timePoint in reversed(stockMatrix.T):
        intrinsicRow = [putCall(S, strike) for S in timePoint]
        intrinsicM[:,-count] = intrinsicRow
        count+=1
    return intrinsicM


#regression
def design(X):
    """Design Matrix for linear regression"""
    basis1 = np.exp(-X/2)
    basis2 = np.exp(-X/2)*(1-X)
    basis3 = np.exp(-X/2)*(1-2*X+X**2/2)
    cov = np.concatenate((basis1, basis2, basis3)).T
    return cov

#cashflow matrix
def cashflow(spot, r, vol, timePointsYear, strike, T, n):
    """Calculate the cashflow matrix based on the optimal stopping rule by lsm algorithm"""
    timePoints = T * timePointsYear
    intrinsicM = intrinsic(spot, r, vol, timePointsYear, strike, T, n)
    #Watch out for changed dimension of cashflow and stoppingRule
    cashFlow = np.zeros((n, timePoints))
    stoppingRule = np.zeros((n, timePoints))
    
    count = 1 # number of timesteps taken (remember backward induction, hence starting at 1)
    for timePoint in reversed(stockMatrix.T):
        if (count == 1):
            stoppingRule[:, -count] = 1
            cashFlow[:, -count] = intrinsicM[:, -count] 
        else:
            intrinsicColumn = intrinsicM[:, -count]
            X = np.array([timePoint[intrinsicColumn>0]])
            numInMoney = np.sum(intrinsicColumn>0) #number of in money paths
            Y = np.zeros(numInMoney)
            index = 0
            for i in range(len(timePoint)):
                if (intrinsicColumn[i]>0):
                    for k in range(timePoints-count+1, timePoints):
                        if stoppingRule[i,k]==1:
                            Y[index] = cashFlow[i,k]*np.exp(-r*(k-(timePoints-count+1)))
                            index +=1
                else:
                    pass
            #TODO:look here, I think X and Y is right now
            designM = design(X)
            lin_reg=LinearRegression()
            regPar = lin_reg.fit(designM,Y) 
            condExp = regPar.intercept_ + np.matmul(designM, regPar.coef_)
            intrinsicVal = intrinsicColumn[intrinsicColumn>0] 
            q = -1 # q counter
            for j in range(len(intrinsicColumn)):
                if (intrinsicColumn[j]>0):
                    q+=1
                    if (condExp[q] > intrinsicVal[q]):
                        pass
                    else:
                        stoppingRule[j, timePoints - count] = 1
                        cashFlow[j, timePoints - count] = intrinsicColumn[j]
                        for k in range(count-1):
                            stoppingRule[j, timePoints-(count-1)+k]=0
                            cashFlow[j, timePoints-(count-1)+k]=0
                else:
                    pass
        count+=1        
    return cashFlow

