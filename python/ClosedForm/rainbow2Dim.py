#article http://finmod.co.za/Pricing%20Rainbow%20Options.pdf
#rainbow options
import numpy as np
from scipy.stats import multivariate_normal

def varIJ(i,j, vol):
    return vol[i]**2 + vol[j]**2 - 2*corr*vol[i]*vol[j]

def d1Prime(i,j, S, vol):
    #d'+
    num = np.log(S[i]/S[j])+(varIJ(i,j)/2)*T
    denom = np.sqrt(varIJ(i,j, vol)*T)
    return num/denom
    
def d2Prime(i,j, S, vol):
    #d'-
    num = np.log(S[i]/S[j])-(varIJ(i,j, vol)/2)*T
    denom = np.sqrt(varIJ(i,j, vol)*T)
    return num/denom


def d1(i, S, vol):
    #d+
    num = np.log(S[i]/K)+(r+vol[i]**2/2)*T
    denom = vol[i]*np.sqrt(T)
    return num/denom

def d2(i, S, vol):
    #d-
    num = np.log(S[i]/K)+(r-vol[i]**2/2)*T
    denom = vol[i]*np.sqrt(T)
    return num/denom

def rho(i,j,k,vol):
    num = corr * vol[i]*vol[j] - corr * vol[i] * vol[k] - corr * vol[k] * vol[j] + vol[k]**2
    denom = np.sqrt((vol[i]**2+vol[k]**2-2*corr*vol[i]*vol[k])*(vol[j]**2+vol[k]**2-2*corr*vol[j]*vol[k]))
    return num/denom

########
# maximum of two assets
#########
d=2
S0 = 40
S1 = 40
vol1 = 0.2
vol2 = 0.3
vol3 = 0
corr = 0.5
r = 0.06
T = 1
K = 40



def callMax2(S0,S1, K, r, T, corr, vol1, vol2, vol3):
    vol = [vol1,vol2,vol3]
    S = [S0,S1]
    corr1 = np.array([ [1, rho(1,2,0, vol)],[rho(1,2,0,vol),1]])
    corr2 = np.array([ [1, rho(0,2,1,vol)],[rho(0,2,1,vol),1]])
    corr3 = np.array([[1,corr],[corr,1]])
    cmax = 0
    cmax += S[0]*multivariate_normal.cdf(x = np.array([-d2Prime(1,0, S, vol),d1(0, S, vol)]), cov=corr1)
    cmax += S[1]*multivariate_normal.cdf(x = np.array([-d2Prime(0,1, S, vol),d1(1, S, vol)]), cov=corr2)
    cmax -= K*np.exp(-r*T)*(1-multivariate_normal.cdf(x = np.array([-d2(0, S, vol),-d2(1, S, vol)]), cov=corr3))
    return cmax
#print("callMax 2 dim: ", callMax2(S0,S1,K,r,T,corr, vol1,vol2,vol3))

#Nice relationship Johnson
import closedEuro
callMIN =  closedEuro.priceECall(t=0,s=S0,sigma=vol1,K=K,r=r,T=T) + closedEuro.priceECall(t=0,s=S1,sigma=vol2,K=K,r=r,T=T) - callMax2(S0,S1,K,r,T,corr, vol1,vol2,vol3)
#print("call min 2 dim: ", callMIN)


#Ouwehand article

def callMin2(S0,S1, K, r, T, corr, vol1, vol2, vol3):
    vol = [vol1,vol2,vol3]
    S = [S0,S1]
    corr1 = np.array([ [1, -rho(1,2,0, vol)],[-rho(1,2,0, vol),1]])
    corr2 = np.array([ [1, -rho(0,2,1, vol)],[-rho(0,2,1,vol),1]])
    corr3 = np.array([[1,corr],[corr,1]])
    cmin = 0
    cmin += S[0]*multivariate_normal.cdf(x = np.array([d2Prime(1,0, S, vol),d1(0, S, vol)]), cov=corr1)
    cmin += S[1]*multivariate_normal.cdf(x = np.array([d2Prime(0,1, S, vol),d1(1, S, vol)]), cov=corr2)
    cmin -= K*np.exp(-r*T)*(multivariate_normal.cdf(x = np.array([d2(0, S, vol),d2(1, S, vol)]), cov=corr3))
    return cmin
    
#No strike callmin
def callMin2NoStrike(S0,S1, r, T, vol1,vol2,vol3):
    vol = [vol1,vol2,vol3]
    S = [S0,S1]
    cmin = 0
    cmin += S[0]*multivariate_normal.cdf(x = d2Prime(1,0, S, vol))
    cmin += S[1]*multivariate_normal.cdf(x = d2Prime(0,1, S, vol))
    return cmin

########
# find european put minimum on two assets
#######
#put-Call parity
def putMin2(S0,S1,K,r,T,corr,vol1,vol2):
    vol3=0
    vol = [vol1,vol2,vol3]
    S = [S0,S1]
    return callMin2(S0, S1,K,r,T,corr,vol1,vol2, vol3) - callMin2NoStrike(S0,S1,r,T, vol1, vol2, vol3) + K*np.exp(-r*T)

#print(putMin2(S0,S1,K,r,T,corr,vol1,vol2,vol3))