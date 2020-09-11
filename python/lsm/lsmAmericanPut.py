import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import simPathStock

#Variables for american put
spot=36
r=0.06
vol=0.2
timePointsYear=5
T=1
n= 10**1
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

#contract function + exercise value 2d numpy array
putCall = lambda S, K: K-S if (K > S) else 0 
putCall_vec = np.vectorize(putCall)
intrinsic = putCall_vec(S,40.0)
CFL =intrinsic[:,1:]

XputCall = lambda S, K: S if (K > S) else 0 
XputCall_vec = np.vectorize(XputCall)
X = XputCall_vec(S,40.0)
Xsh = np.delete(arr=X,obj=steps,axis=1)
X2sh = np.multiply(Xsh,Xsh)
X2sh

Y1 = intrinsic*np.exp(-1*r*(T/steps))
Y2 = np.concatenate((np.zeros((n,steps-1)),Y1[:,steps:]),axis=1)

CV = np.zeros((n,steps-1))

from sklearn.linear_model import LinearRegression

for i in range(steps-1,0,-1):
    coef = np.array((Xsh[:,i-1], X2sh[:,i-1])).T
    reg1 = LinearRegression().fit(X=coef,y=Y2[:,i:])
    CV[:,i-1:] = reg1.predict(coef)
    for j in range(len(intrinsic)):
        if intrinsic[j,i-1]>CV[j,i-1]:
            Y2[j,i-1] = Y1[j,i-1] 
        else:
            Y2[j,i-1] = Y2[j,i]*np.exp(-1*r*(T/steps))
    
CVp = np.concatenate((CV, np.zeros((n,1))),axis=1)
POF = (0 if CVp>CFL else CFL)

############
## data
############




#regression
def design(X, choice):
    if choice==1:
        "Design Matrix for linear regression"""
        basis1 = np.exp(-X/2)
        basis2 = np.exp(-X/2)*(1-X)
        basis3 = np.exp(-X/2)*(1-2*X+X**2/2)
        cov = np.concatenate((basis1, basis2, basis3)).T
    elif choice==2:
        """Design Matrix for linear regression"""
        basis1 = X
        basis2 = X**2
        cov = np.concatenate((basis1, basis2)).T
    else:
        choice = input("You entered an invalid choice for a basis, please enter 1 or 2")
        return design(X, choice)
    return cov

#cashflow matrix
def cashflow(spot, r, vol, timePointsYear, strike, T, n, choice, stockMatrix):
    """Calculate the cashflow matrix based on the optimal stopping rule by lsm algorithm"""
    timePoints = T * timePointsYear
    intrinsicM = intrinsic(spot, timePointsYear, strike, T, n, stockMatrix)
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
            designM = design(X, choice)
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

cashFlowMatrix = cashflow(spot=36, r=0.06, vol=0.2, timePointsYear=50, strike=40, T=1, n=n, choice=1, stockMatrix=S)

def findPV(r, cashFlowMatrix, timePointsYear):
    """Find present value of a cashflow matrix starting at 1. timestep"""
    PV = 0
    timeSteps = 1/timePointsYear
    for i in range(cashFlowMatrix.shape[0]):
        for j in range(cashFlowMatrix.shape[1]):
            PV += cashFlowMatrix[i,j]*np.exp(-r*((j+1)*timeSteps))
    return (PV/cashFlowMatrix.shape[0])
    
print(findPV(r, cashFlowMatrix, timePointsYear))



