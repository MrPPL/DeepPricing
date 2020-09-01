#read file
import sys 
import os
sys.path.append(os.path.abspath("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/python/ClosedForm"))
import closedEuro
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
    K=100 #sets strike to 100
    for row in X:
        Y[count] = closedEuro.priceECall(t=0,s=row[0]*K,sigma=row[3],K=100,r=row[2], T=row[1])/K
        count+=1
    return Y


#parameter ranges for european options
moneyness = (0.8,1.2)
T = (1/252,3.0)
r = (0.01,0.03)
vol = (0.05,0.5)
parVec = [moneyness,T,r,vol]

# number of samples
nSamp=3 * 10**5
#make data
X = quasiSampling(parVec, nSamp)
Y = findY(X)

import pandas as pd
df = pd.DataFrame({'y':Y[:,0],'moneyness':X[:,0],'T':X[:,1],'r':X[:,2], 'vol':X[:,3]})
df.to_csv("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/python/deepLearning/hirsa19/data/mediumCEuroData.csv")


    