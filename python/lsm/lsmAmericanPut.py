import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import simPathStock

#Variables for american put
spot=1
r=0.06
vol=0.4
timePointsYear=3
T=1
n=8
strike = 1.1

#stockMatrix

path1 = [1,1.09,1.08,1.34]
path2 = [1,1.16,1.26,1.54]
path3 = [1,1.22,1.07,1.03]
path4 = [1,0.93,0.97,0.92]
path5 = [1,1.11,1.56,1.52]
path6 = [1,0.76,0.77,0.90]
path7 = [1,0.92,0.84,1.01]
path8 = [1,0.88,1.22,1.34]

stockMatrix = np.array([path1,path2, path3, path4, path5, path6, path7, path8])


#contract function
putCall = lambda S, K: K-S if (K > S) else 0 

#Intrinsic matrix
def intrinsic(spot, timePointsYear, strike, T, n):
    """ Finds the intrinsic value matrix """
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
def design1(X):
    """Design Matrix for linear regression"""
    basis1 = X
    basis2 = X**2
    cov = np.concatenate((basis1, basis2)).T
    return cov

#cashflow matrix
def cashflow(spot, r, vol, timePointsYear, strike, T, n):
    """Calculate the cashflow matrix based on the optimal stopping rule by lsm algorithm"""
    timePoints = T * timePointsYear
    intrinsicM = intrinsic(spot, timePointsYear, strike, T, n)
    #Watch out for changed dimension of cashflow and stoppingRule
    cashFlow = np.zeros((n, timePoints))
    stoppingRule = np.zeros((n, timePoints))
    
    count = 1 # number of timesteps taken (remember backward induction, hence starting at 1)
    for timePoint in reversed(stockMatrix.T):
        if (count == 1):
            stoppingRule[:, -count] = 1
            cashFlow[:, -count] = intrinsicM[:, -count] 
        elif (count == timePoints+1):
            #can only exercise after initiazation of option
            break
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
            designM = design1(X)
            lin_reg=LinearRegression()
            regPar = lin_reg.fit(designM,Y) 
            condExp = regPar.intercept_ + np.matmul(designM, regPar.coef_)
            intrinsicVal = intrinsicColumn[intrinsicColumn>0] 
            q = -1 # q counter
            for j in range(len(intrinsicColumn)):
                if (intrinsicColumn[j]>0):
                    q+=1
                    if (condExp[q] <= intrinsicVal[q]):
                        stoppingRule[j, timePoints - count] = 1
                        cashFlow[j, timePoints - count] = intrinsicColumn[j]
                        for k in range(count-1):
                            stoppingRule[j, timePoints-(count-1)+k]=0
                            cashFlow[j, timePoints-(count-1)+k]=0
        count+=1  
    
    return (cashFlow)



def findPV(spot, r, vol, timePointsYear, strike, T, n):
    cashFlowMatrix = cashflow(spot, r, vol, timePointsYear, strike, T, n)
    PV = 0
    for i in range(cashFlowMatrix.shape[0]):
        for j in range(cashFlowMatrix.shape[1]):
            PV += cashFlowMatrix[i,j]*np.exp(-r*(j+1))
    return (PV/cashFlowMatrix.shape[0])
    
print(findPV(spot, r, vol, timePointsYear, strike, T, n))




