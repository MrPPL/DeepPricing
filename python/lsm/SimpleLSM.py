"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           Monte Carlo simulation of stock with constant volatility and risk free rate
// Course:          Master thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
// Encoding:        utf-8
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Options, Futures, and Other Derivatives by John C. Hull 10th edition
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
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
        
                

             
        


#make simple example lsm

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
def intrinsic(spot, r, timePointsYear, strike, T, n):
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
    basis1 = X
    basis2 = X**2
    cov = np.concatenate((basis1, basis2)).T
    return cov

#cashflow matrix
def cashflow(spot, r, timePointsYear, strike, T, n):
    """Calculate the cashflow matrix based on the optimal stopping rule by lsm algorithm"""
    timePoints = T * timePointsYear
    intrinsicM = intrinsic(spot, r, timePointsYear, strike, T, n)
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


print(cashflow(spot=1, r=0.06, timePointsYear=3, strike=1.1, T=1, n=8))




#Basis functions for regression
basis1 = lambda X: np.exp(-X/2)
basis2 = lambda X: np.exp(-X/2)*(1-X)
basis3 = lambda X: np.exp(-X/2)(1-2*X+X^2/2)

#Cashflow and stopping rule matrices
timePoints = T * timePointsYear
cashFlow = np.zeros((n, timePoints))
stoppingRule = np.zeros((n, timePoints))
for timePoint in stockMatrix:
    print(timePoint)


def findCashFlow(simPath, K, r):
    """ Given a df of simulated paths (simpath), the strike K (a number) 
    and the interest rate r (a number), the function finds the optimal stopping
    stragedy for a american put option. The output is a cashflow df"""
    inMoneyMatrix = simPath.iloc[:,1:].applymap(lambda x: max(K-x, 0))
    cashFlowMatrix = np.zeros((simPath.shape[0],simPath.shape[1]-1))
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



def findPV(cashFlowMatrix, r):
    count=1 
    PVMatrix = np.zeros((cashFlowMatrix.shape[0],cashFlowMatrix.shape[1]))
    for x in cashFlowMatrix:
        PVMatrix[:,count-1] = cashFlowMatrix[x]*np.exp(-r*count)
        count+=1
    PV = sum(np.sum(PVMatrix,axis=1))/cashFlowMatrix.shape[0]
    return PV
    

print(findPV(cashFlow, 0.06))
        
                

             
        



#Basis functions for regression
basis1 = lambda X: np.exp(-X/2)
basis2 = lambda X: np.exp(-X/2)*(1-X)
basis3 = lambda X: np.exp(-X/2)(1-2*X+X^2/2)

#Cashflow and stopping rule matrices
timePoints = T * timePointsYear
cashFlow = np.zeros((n, timePoints))
stoppingRule = np.zeros((n, timePoints))
for timePoint in stockMatrix:
    print(timePoint)


def findCashFlow(simPath, K, r):
    """ Given a df of simulated paths (simpath), the strike K (a number) 
    and the interest rate r (a number), the function finds the optimal stopping
    stragedy for a american put option. The output is a cashflow df"""
    inMoneyMatrix = simPath.iloc[:,1:].applymap(lambda x: max(K-x, 0))
    cashFlowMatrix = np.zeros((simPath.shape[0],simPath.shape[1]-1))
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



def findPV(cashFlowMatrix, r):
    count=1 
    PVMatrix = np.zeros((cashFlowMatrix.shape[0],cashFlowMatrix.shape[1]))
    for x in cashFlowMatrix:
        PVMatrix[:,count-1] = cashFlowMatrix[x]*np.exp(-r*count)
        count+=1
    PV = sum(np.sum(PVMatrix,axis=1))/cashFlowMatrix.shape[0]
    return PV
    

print(findPV(cashFlow, 0.06))
        
                

             
        



#Basis functions for regression
basis1 = lambda X: np.exp(-X/2)
basis2 = lambda X: np.exp(-X/2)*(1-X)
basis3 = lambda X: np.exp(-X/2)(1-2*X+X^2/2)

#Cashflow and stopping rule matrices
timePoints = T * timePointsYear
cashFlow = np.zeros((n, timePoints))
stoppingRule = np.zeros((n, timePoints))
for timePoint in stockMatrix:
    print(timePoint)


def findCashFlow(simPath, K, r):
    """ Given a df of simulated paths (simpath), the strike K (a number) 
    and the interest rate r (a number), the function finds the optimal stopping
    stragedy for a american put option. The output is a cashflow df"""
    inMoneyMatrix = simPath.iloc[:,1:].applymap(lambda x: max(K-x, 0))
    cashFlowMatrix = np.zeros((simPath.shape[0],simPath.shape[1]-1))
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



def findPV(cashFlowMatrix, r):
    count=1 
    PVMatrix = np.zeros((cashFlowMatrix.shape[0],cashFlowMatrix.shape[1]))
    for x in cashFlowMatrix:
        PVMatrix[:,count-1] = cashFlowMatrix[x]*np.exp(-r*count)
        count+=1
    PV = sum(np.sum(PVMatrix,axis=1))/cashFlowMatrix.shape[0]
    return PV
    

print(findPV(cashFlow, 0.06))
        
                

             
        
