#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:46:32 2020

@author: ppl
"""

import numpy as np
import pandas as pd
# import model
from sklearn.linear_model import LinearRegression


#make simple example lsm

path1 = [1,1.09,1.08,1.34]
path2 = [1,1.16,1.26,1.54]
path3 = [1,1.22,1.07,1.03]
path4 = [1,0.93,0.97,0.92]
path5 = [1,1.11,1.56,1.52]
path6 = [1,0.76,0.77,0.90]
path7 = [1,0.92,0.84,1.01]
path8 = [1,0.88,1.22,1.34]

stockMatrix = pd.DataFrame(np.array([path1,path2, path3, path4, path5, path6, path7, path8]))

## variables
r=0.06






def findCashFlow(simPath, K, r):
    """ Takes in simulated paths of e.g. stock (type df), calculate the optimal cashflow of the option 
    given a strike K (type number) and a interest rate r (type number)"""
    
    inMoneyMatrix = stockMatrix.iloc[:,1:].applymap(lambda x: max(K-x, 0))
    cashFlowMatrix = np.zeros((stockMatrix.shape[0],stockMatrix.shape[1]-1))
    stopRule = np.zeros((stockMatrix.shape[0],stockMatrix.shape[1]-1))
    for j in reversed(range(cashFlowMatrix.shape[1])):
        # initial point
        if (j == (cashFlowMatrix.shape[1]-1)):
            cashFlowMatrix[:,j] = inMoneyMatrix.values[:,j]
            stopRule[:,j] = np.multiply(inMoneyMatrix.values[:,j]>0, 1)
        else:
            X = np.zeros((sum(inMoneyMatrix.values[:,j]>0),1))
            Y = np.zeros(sum(inMoneyMatrix.values[:,j]>0))
            count=0
            for i in range(stockMatrix.shape[0]):
                if (inMoneyMatrix.values[i,j]>0):
                    X[count]=stockMatrix.values[i,j+1]
                    # find discount factor
                    for k in range(stockMatrix.shape[1]-1-j):
                        if (stopRule[i,j+k]==1):
                            Y[count]= cashFlowMatrix[i,j+k]*np.exp(-r*k)
                    count+=1
            # Linear regression
            basis = X
            basis1 = X**2
            cov = np.concatenate((basis, basis1), axis=1)
            lin_reg=LinearRegression()
            lin_reg.fit(cov,Y)
        
            # output and find stopMatrix + cashflow matrix
            regressVec = np.zeros(sum(inMoneyMatrix.values[:,j]>0))
            count = 0
            for i in range(stockMatrix.shape[0]):
                if (inMoneyMatrix.values[i,j]>0):
                    regressVec[count] =  lin_reg.intercept_ + lin_reg.coef_[0]*X[count] + lin_reg.coef_[1]*X[count]**2
                    if (inMoneyMatrix.values[i,j]>regressVec[count]):
                        stopRule[i,:] = np.zeros(inMoneyMatrix.shape[1])
                        stopRule[i,j] = 1
                        cashFlowMatrix[i,:] = np.zeros(inMoneyMatrix.shape[1])
                        cashFlowMatrix[i,j] = inMoneyMatrix.values[i,j]
                    count+=1
    return cashFlowMatrix

cashFlow = pd.DataFrame(findCashFlow(stockMatrix, 1.1,0.06))



def findPV(cashFlowMatrix):
    count=1 
    PVMatrix = np.zeros((cashFlowMatrix.shape[0],cashFlowMatrix.shape[1]))
    for x in cashFlowMatrix:
        PVMatrix[:,count-1] = cashFlowMatrix[x]*np.exp(-r*count)
        count+=1
    PV = sum(np.sum(PVMatrix,axis=1))/cashFlowMatrix.shape[0]
    return PV
    

print(findPV(cashFlow))
        
                

             
        

