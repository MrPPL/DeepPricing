#article http://finmod.co.za/Pricing%20Rainbow%20Options.pdf
#rainbow call on the max option
import numpy as np
from scipy.stats import multivariate_normal

def varIJ(i,j):
    return vol[i]**2 + vol[j]**2 - 2*corr*vol[i]*vol[j]

def d1Prime(i,j):
    #d'+
    num = np.log(S[i]/S[j])+(varIJ(i,j)/2)*T
    denom = np.sqrt(varIJ(i,j)*T)
    return num/denom
    
def d2Prime(i,j):
    #d'-
    num = np.log(S[i]/S[j])-(varIJ(i,j)/2)*T
    denom = np.sqrt(varIJ(i,j)*T)
    return num/denom


def d1(i):
    #d+
    num = np.log(S[i]/K)+(r+vol[i]**2/2)*T
    denom = vol[i]*np.sqrt(T)
    return num/denom

def d2(i):
    #d-
    num = np.log(S[i]/K)+(r-vol[i]**2/2)*T
    denom = vol[i]*np.sqrt(T)
    return num/denom

def rho(i,j,k):
    num = corr * vol[i]*vol[j] - corr * vol[i] * vol[k] - corr * vol[k] * vol[j] + vol[k]**2
    denom = np.sqrt((vol[i]**2+vol[k]**2-2*corr*vol[i]*vol[k])*(vol[j]**2+vol[k]**2-2*corr*vol[j]*vol[k]))
    return num/denom

#European Call on the maximum of two assets
S = [10,10,10]
vol = [0.2,0.2,0.2,0.2]
corr = 0.1
r = 0.1
T = 1
K = 10
d=3


#make correlation matrix
#make correlation matrix

vol1=1
lowerpart = 0.5
lowerpart1 = 0.1
corrMatrix1 = np.array([ [vol1**2, rho(1,2,0),rho(1,3,0)],[lowerpart,vol1**2,rho(2,3,0)],[lowerpart,lowerpart,vol1**2]])
corrMatrix2 = np.array([[vol1**2,corr,corr],[lowerpart1,vol1**2,corr],[lowerpart1,lowerpart1,vol1**2]])
corrMatrix1
corrMatrix2

cov1 = [[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]]
cov2 = [[1,0.1,0.1],[0.1,1,0.1],[0.1,0.1,1]] 

cmax = 0
cmax += S[0]*multivariate_normal.cdf(x = np.array([-d2Prime(1,0),-d2Prime(2,0),d1(0)]), cov=corrMatrix1)
cmax += S[1]*multivariate_normal.cdf(x = np.array([-d2Prime(0,1),-d2Prime(2,1),d1(1)]), cov=corrMatrix1)
cmax += S[2]*multivariate_normal.cdf(x = np.array([-d2Prime(0,2),-d2Prime(1,2),d1(2)]), cov=corrMatrix1)
cmax -= K*np.exp(-r*T)*(1-multivariate_normal.cdf(x = np.array([-d2(0),-d2(1),-d2(2)]), cov=corrMatrix2))
print(cmax)

#BEG examples
#make correlation matrix
#make correlation matrix
#European Call on the maximum of two assets
S = [100,100,100]
vol = [0.2,0.2,0.2,0.2]
corr = 0.5
r = 0.1
T = 1
K = 100
d=3


corrMatrix1 = np.array([ [1, rho(1,2,0),rho(1,3,0)],[rho(1,2,0),1,rho(2,3,0)],[rho(1,3,0),rho(2,3,0),1]])
corrMatrix2 = np.array([[1,corr,corr],[corr,1,corr],[corr,corr,1]])
corrMatrix1
corrMatrix2

# maximum of assets
cmax = 0
cmax += S[0]*multivariate_normal.cdf(x = np.array([-d2Prime(1,0),-d2Prime(2,0),d1(0)]), cov=corrMatrix1)
cmax += S[1]*multivariate_normal.cdf(x = np.array([-d2Prime(0,1),-d2Prime(2,1),d1(1)]), cov=corrMatrix1)
cmax += S[2]*multivariate_normal.cdf(x = np.array([-d2Prime(0,2),-d2Prime(1,2),d1(2)]), cov=corrMatrix1)
cmax -= K*np.exp(-r*T)*(1-multivariate_normal.cdf(x = np.array([-d2(0),-d2(1),-d2(2)]), cov=corrMatrix2))
print(cmax)
