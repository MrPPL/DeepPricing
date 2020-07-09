#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:34:44 2020

@author: ppl
"""

import numpy as np
import pandas as pd


# LSM algorihmn
#%%
#Simulate paths
sigma = 0.2 # Standard deviation
S = 36 #present stock price
r=0.06 #risk free rate
K=40 # strike price
L = 50 # discrete points
T=1 # number of years

n = 1000 # number of paths half of total paths see antithetic

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
    simPaths= pd.concat([dat1, dat2], axis=1) #return pandas dataframe
    return (simPaths)

simStockPath(n=2, T=1, L=50, S=36, r=0.06, sigma=0.2)

#!!!!!!!!!!!!!!! OBS : Antithetic !!!!!!!!!!!!!!!!!!!!

#%%
#Evaluate contract
stockPaths = simStockPath(n=2, T=1, L=50, S=36, r=0.06, sigma=0.2)
exerciseValue = stockPaths.applymap(lambda x: max(40-x,0)) #apply func to every element of DF

stockPaths.shape

#%%
#Linear regression
from sklearn.linear_model import LinearRegression

type(stockPaths)
type(exerciseValue)

stockPaths.shape

stockPaths

# Laguerre polynomials
basis1 = stockPaths.applymap(lambda x: np.exp(-x/2))
basis2 = stockPaths.applymap(lambda x: np.exp(-x/2) *(1-x))
basis3 = stockPaths.applymap(lambda x: np.exp(-x/2) *(1-2*x+x**2/2))

exerciseValue
exerciseValue.shape[0]
exerciseValue.iloc[50,:]
sum(exerciseValue.iloc[50,:]>0)


#Linear regresion
def findY(L,T, n,r):
    points = L*T #number of discrete point
    stopRule = np.zeros( (points+1, n) ) # matrix for stopping rule
    cashFlowMatrix = np.zeros( (points+1, n) ) # matrix for cashflow
    for i in reversed(range(0, exerciseValue.shape[0])):
        #last row of cashFlowMatrix and stopRuleMatrix
        if (i == (exerciseValue.shape[0]-1)):
            cashFlowMatrix[i,:] = exerciseValue.iloc[i,:]
            for j in range(exerciseValue.shape[1]):
                if (cashFlowMatrix[i,j]>0):
                    stopRule[i,j]=1
        
        #Regression
        if (i == (exerciseValue.shape[0]-2)):
            X = stockPaths.iloc[i,:][ exerciseValue.iloc[i,:]>0 ] #
            Y = exerciseValue.iloc[i+1,:][ exerciseValue.iloc[i,:]>0 ] * np.exp(-r)
            #instantiate
            linReg = LinearRegression(fit_intercept=True) # object
            
            basis1 = lambda x: np.exp(-x/2)
            basis2 = lambda x: np.exp(-x/2) *(1-x)
            basis3 = lambda x: np.exp(-x/2) *(1-2*x+x**2/2)
            cov = pd.DataFrame({'Basis1': basis1(X) , 'Basis2': basis2(X), 'Basis3': basis3(X)})
            linReg.fit(cov,y) 
            #estimate
            #print estimate
            print(linReg.intercept_)
            print(linReg.coef_)
            
        else:
            pass
     
    return 
        
        
    #     numPath = sum(exerciseValue.iloc[i,:]>0) #number of paths in the money
    #     continuation = pd.Series([1,2,3])
    #     continuation
    #     type(continuation)
    #     count = 0
        
    #     for j in range(exerciseValue.shape[1]):
    #         if (exerciseValue.iloc[i,j]>0):
    #             inMoneyExer[count] = exerciseValue.iloc[i,j]
    #             count = count + 1 
    # return inMoneyExer

print(findY(sigma = 0.2, S = 36, r=0.06, K=40, L = 50 ))
      
n = 1000 # number of paths half of total paths see antithetic))
    


X = pd.DataFrame({'Basis1': basis1.iloc[:,1] , 'Basis2': basis2.iloc[:,1], 'Basis3': basis3.iloc[:,1]})
y = exerciseValue.iloc[:,1]
X.shape
y.shape
#instantiate
linReg = LinearRegression(fit_intercept=True) # object
linReg.fit(X,y) 
#print estimate
print(linReg.intercept_)
print(linReg.coef_)
#
feature_cols = ["Basis1", "Basis2", "Basis3"]
zip(feature_cols, linReg.coef_)

#%%
#Plotting
# conventional way to import seaborn
import seaborn as sns
data= pd.concat([y, X], axis=1)

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data=data,x_vars=feature_cols, y_vars=1)

#%%
for i in reversed(range(0, exerciseValue.shape[0])):
    print(i)


