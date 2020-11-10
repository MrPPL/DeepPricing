#read file
import sys 
import os
sys.path.append(os.path.abspath("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/python/ClosedForm"))
import rainbow2Dim
import torch

#Create data
import numpy as np
#halton sequence
#quasi approach
def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True
 
    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return np.array(seq)


#create simulation by halton*range+lower bound
def quasiSampling(parVec, samples):
    samp = halton_sequence(samples,len(parVec)).T
    count = 0
    for par in parVec:
        samp[:,count] = samp[:,count] * (par[1]-par[0]) + par[0]
        count += 1
    return samp

def findY(X):
    Y = np.empty((X.shape[0],1))
    count=0
    for row in X:
        Y[count] = rainbow2Dim.putMin2(S0=row[0], S1=row[1], K=row[2], r=row[4], T=row[3], corr = row[7], vol1=row[5], vol2=row[6]) 
        count+=1
    return Y


#parameter ranges for european options
moneyness1 = (0.8,1.2)
moneyness2 = (0.8,1.2)
T = (1/252,3.0)
r = (0.01,0.03)
vol1 = (0.05,0.5)
vol2 = (0.05,0.5)
rho = (-0.5,0.5)
parVec = [moneyness1, moneyness2, T, r, vol1, vol2, rho]

# number of samples
nSamp= 1* 10**3
#make data
X = quasiSampling(parVec, nSamp)
Y = findY(X)

import pandas as pd
df = pd.DataFrame({'y':Y[:,0],'moneyness1':X[:,0],'moneyness2':X[:,1], 'T':X[:,2],'r':X[:,3], 'vol1':X[:,4], 'vol2':X[:,5], 'rho':X[:,6]})
df.to_csv("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepLearning/hirsa19/data/1KEuroMinPut.csv")


    