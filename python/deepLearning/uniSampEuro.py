
#read file
import sys 
import os
sys.path.append(os.path.abspath("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/python/ClosedForm"))
import closedEuro
import torch

#Create data
import numpy as np

#create simulation by halton*range+lower bound
def unifSampling(parVec, samples):
    samp = np.random.uniform(0,1, size=(nSamp,len(parVec)))
    count = 0
    for par in parVec:
        samp[:,count] = samp[:,count] * (par[1]-par[0]) + par[0]
        count += 1
    return samp

def findY(X):
    Y = np.empty((X.shape[0],1))
    count=0
    for row in X:
        Y[count] = closedEuro.priceECall(t=0,s=row[0],sigma=row[3],K=1,r=row[2], T=row[1])
        count+=1
    return Y


#parameter ranges for european options
moneyness = (0.8,1.2)
T = (1/252,3.0)
r = (0.01,0.03)
vol = (0.05,0.5)
parVec = [moneyness,T,r,vol]

# number of samples
nSamp= 6*10**4
#make data
X = unifSampling(parVec, nSamp)
Y = findY(X)

import pandas as pd
df = pd.DataFrame({'y':Y[:,0],'moneyness':X[:,0],'T':X[:,1],'r':X[:,2], 'vol':X[:,3]})
df.to_csv("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/python/deepLearning/hirsa19/data/uniCEuroData.csv")


    