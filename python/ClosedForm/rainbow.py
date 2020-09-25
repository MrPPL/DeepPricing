#article http://finmod.co.za/Pricing%20Rainbow%20Options.pdf
#rainbow options
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

#BEG examples

#European Call on the maximum of two asset

#Parameters
d=3
S = [100]*d
vol = [0.2]*(d+1)
corr = 0.5
r = 0.1
T = 1
K = 100


corr1 = np.array([ [1, rho(1,2,0),rho(1,3,0)],[rho(1,2,0),1,rho(2,3,0)],[rho(1,3,0),rho(2,3,0),1]])
corr2 = np.array([[1,corr,corr],[corr,1,corr],[corr,corr,1]])
#############
# maximum of three assets
########
def callMax3(S, K, r, T, corr1, corr2):
    cmax = 0
    cmax += S[0]*multivariate_normal.cdf(x = np.array([-d2Prime(1,0),-d2Prime(2,0),d1(0)]), cov=corr1)
    cmax += S[1]*multivariate_normal.cdf(x = np.array([-d2Prime(0,1),-d2Prime(2,1),d1(1)]), cov=corr1)
    cmax += S[2]*multivariate_normal.cdf(x = np.array([-d2Prime(0,2),-d2Prime(1,2),d1(2)]), cov=corr1)
    cmax -= K*np.exp(-r*T)*(1-multivariate_normal.cdf(x = np.array([-d2(0),-d2(1),-d2(2)]), cov=corr2))
    return cmax

corrMin1 = np.array([ [1, rho(1,2,0),-rho(1,3,0)],[rho(1,2,0),1,-rho(2,3,0)],[-rho(1,3,0),-rho(2,3,0),1]])
corr2 = np.array([[1,corr,corr],[corr,1,corr],[corr,corr,1]])
def callMin3(S, K, r, T, corr1, corr2):
    cmin = 0
    cmin += S[0]*multivariate_normal.cdf(x = np.array([d2Prime(1,0),d2Prime(2,0),d1(0)]), cov=corr1)
    cmin += S[1]*multivariate_normal.cdf(x = np.array([d2Prime(0,1),d2Prime(2,1),d1(1)]), cov=corr1)
    cmin += S[2]*multivariate_normal.cdf(x = np.array([d2Prime(0,2),d2Prime(1,2),d1(2)]), cov=corr1)
    cmin -= K*np.exp(-r*T)*(multivariate_normal.cdf(x = np.array([d2(0),d2(1),d2(2)]), cov=corr2))
    return cmin

print(callMax3(S,K,r,T,corr1,corr2))
print(callMin3(S,K,r,T,corrMin1,corr2))
########
# maximum of two assets
#########
d=2
S = [40]*d
vol = [0.2,0.3, 0]
corr = 0.5
r = 0.06
T = 1
K = 40


corr1 = np.array([ [1, rho(1,2,0)],[rho(1,2,0),1]])
corr2 = np.array([ [1, rho(0,2,1)],[rho(0,2,1),1]])
corr3 = np.array([[1,corr],[corr,1]])

def callMax2(S, K, r, T, corr1, corr2, corr3):
    cmax = 0
    cmax += S[0]*multivariate_normal.cdf(x = np.array([-d2Prime(1,0),d1(0)]), cov=corr1)
    cmax += S[1]*multivariate_normal.cdf(x = np.array([-d2Prime(0,1),d1(1)]), cov=corr2)
    cmax -= K*np.exp(-r*T)*(1-multivariate_normal.cdf(x = np.array([-d2(0),-d2(1)]), cov=corr3))
    return cmax
print(callMax2(S,K,r,T,corr1,corr2, corr3))

def callMin2(S, K, r, T, corr1, corr2, corr3):
    cmin = 0
    cmin += S[0]*multivariate_normal.cdf(x = np.array([d2Prime(1,0),d1(0)]), cov=corr1)
    cmin += S[1]*multivariate_normal.cdf(x = np.array([d2Prime(0,1),d1(1)]), cov=corr2)
    cmin -= K*np.exp(-r*T)*(multivariate_normal.cdf(x = np.array([d2(0),d2(1)]), cov=corr3))
    return cmin
print(callMin2(S,K,r,T,corr1,corr2, corr3))

print(callMin2(S,K,r,T,corr1,corr2, corr3) - callMin2(S,0,r,T,corr1,corr2, corr3) + K*np.exp(-r*T))