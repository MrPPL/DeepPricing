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
spot1 = (80,120)
spot2 = (80,120)
K = (100,100)
T = (1/252,3.0)
r = (0.01,0.03)
vol1 = (0.05,0.5)
vol2 = (0.05,0.5)
rho = (0.05,0.5)
parVec = [spot1, spot2, K, T, r, vol1, vol2, rho]

# number of samples
nSamp= 3* 10**5
#make data
X = quasiSampling(parVec, nSamp)
Y = findY(X)

import pandas as pd
df = pd.DataFrame({'y':Y[:,0],'spot1':X[:,0],'spot2':X[:,1], 'K':X[:,2], 'T':X[:,3],'r':X[:,4], 'vol1':X[:,5], 'vol2':X[:,6], 'rho':X[:,7]})
df.to_csv("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/python/deepLearning/hirsa19/data/300KEuroMinPut.csv")


    