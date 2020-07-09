#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:57:55 2020

@author: ppl
"""

import numpy as np
import pandas as pd
# import model
from sklearn.linear_model import LinearRegression

#Simulate paths
sigma = 0.2 # Standard deviation
S = 36 #present stock price
r=0.06 #risk free rate
K=40 # strike price
L = 3 # discrete points
T=1 # number of years

n = 2 # number of paths half of total paths see antithetic

def simStockPath(n, T, L, S, r, sigma):
    points = L*T #number of discrete point
    stepSize = T/points # stepsize of year
    
    paths = np.zeros( (points+1, n) ) # matrix for paths
    paths[0,] = S
    
    antiPaths = np.zeros( (points+1, n) ) # matrix for antithetic paths
    antiPaths[0,] = S
    
    for i in range(1,points+1):
        simW = np.random.normal(loc=0.0, scale=(stepSize)**0.5, size=n)
        for j in range(len(simW)):
            dS = r*paths[i-1,j]*stepSize + sigma*paths[i-1,j]*simW[j]
            paths[i,j] = paths[i-1,j]+dS
            
            dS = r*antiPaths[i-1,j]*stepSize + sigma*antiPaths[i-1,j]*(-simW[j])
            antiPaths[i,j] = antiPaths[i-1,j]+dS          
    dat1 = pd.DataFrame(paths)  
    dat2 = pd.DataFrame(antiPaths)
    simPaths= pd.concat([dat1], axis=1) #return pandas dataframe
    return (simPaths.T)

#%%
np.random.normal(loc=0, scale=4.0, size=10)
2
#!!!!!!!!!!!!!!! OBS : Antithetic !!!!!!!!!!!!!!!!!!!!
#%%
stockMatrix = simStockPath(n=5, T=1, L=4, S=36, r=0.06, sigma=0.2)
stockMatrix
def findCashFlow(simPath, K, r):
    """ Given a df of simulated paths (simpath), the strike K (a number) 
    and the interest rate r (a number), the function finds the optimal stopping
    stragedy for a american put option. The output is a cashflow df"""
    inMoneyMatrix = simPath.iloc[:,1:].applymap(lambda x: max(K-x, 0))
½    cashFlowMatrix = np.zeros((simPath.shape[0],simPath.shape[1]-1))
    stopRule = np.zeros((simPath.shape[0],simPath.shape[1]-1))
    for j in reversed(range(cashFlowMatrix.shape[1])):
        # European option
        if (j == (cashFlowMatrix.shape[1]-1)):
            cashFlowMatrix[:,j] = inMoneyMatrix.values[:,j]
            stopRule[:,j] = np.multiply(inMoneyMatrix.values[:,j]>0, 1)
        else:
            X = np.zeros((sum(inMoneyMatrix.values[:,j]>0),1))
            Y = np.zeros(sum(inMoneyMatrix.values[:,j]>0))
            count=0
            for i in range(simPath.shape[0]):
                if (inMoneyMatrix.values[i,j]>0):
                    X[count]=simPath.values[i,j+1]
                    # find discount factor
                    for k in range(simPath.shape[1]-1-j):
                        if (stopRule[i,j+k]==1):
                            Y[count]= cashFlowMatrix[i,j+k]*np.exp(-r*k)
                    count+=1
            # Linear regression: !!OBS!! basis functions
            basis = np.exp(-X/2)
            basis1 = np.exp(-X/2)*(1-X)
            basis2 = np.exp(-X/2)*(1-2*X+X**2/2)
            cov = np.concatenate((basis, basis1, basis2), axis=1)
            lin_reg=LinearRegression()
            lin_reg.fit(cov,Y)
        
           # decide to exercise or not based on the regression and intrinsic value
            regressVec = np.zeros(sum(inMoneyMatrix.values[:,j]>0))
            count = 0
            for i in range(simPath.shape[0]):
                if (inMoneyMatrix.values[i,j]>0):
                    regressVec[count] =  lin_reg.intercept_ + lin_reg.coef_[0]*X[count] + lin_reg.coef_[1]*X[count]**2
                    if (inMoneyMatrix.values[i,j]>regressVec[count]):
                        stopRule[i,:] = np.zeros(inMoneyMatrix.shape[1])
                        stopRule[i,j] = 1
                        cashFlowMatrix[i,:] = np.zeros(inMoneyMatrix.shape[1])
                        cashFlowMatrix[i,j] = inMoneyMatrix.values[i,j]
                    count+=1
    return pd.DataFrame(cashFlowMatrix)

cashFlow = findCashFlow(stockMatrix, 40,0.06)



def findPV(cashFlowMatrix):
    count=1 
    PVMatrix = np.zeros((cashFlowMatrix.shape[0],cashFlowMatrix.shape[1]))
    for x in cashFlowMatrix:
        PVMatrix[:,count-1] = cashFlowMatrix[x]*np.exp(-r*count)
        count+=1
    PV = sum(np.sum(PVMatrix,axis=1))/cashFlowMatrix.shape[0]
    return PV
    

print(findPV(cashFlow))
        
                

             
        

