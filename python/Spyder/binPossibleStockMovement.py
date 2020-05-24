#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:59:55 2020

@author: ppl
"""
import numpy as np
#just for fun

S=80
periods = 3
K = 80
u=1.5
d=0.5
r=1.1

A=np.zeros(shape=(periods+1,periods+1))
A
for i in range(periods+1):
    for j in range(periods+1):
        if j<periods+1-i:
            A[i,j]=S*d**(j)*u**(periods-i-j)
print(A)


def binPriceCall(t,S, sigma, K, r, T, steps):
    callPrices = np.zeros(shape=(steps+1,steps+1))
    
    u= np.exp(sigma*np.sqrt((T-t)/steps))
    d=1/u
    i=r**((T-t)/steps)
    p= ((1+i)-d)/(u-d)
    
    for i in range(steps+1):
        for j in range(steps+1):
            if i == 0:
                callPrices[i,j] = max(0,u**j*d**(steps-j)*S-K)
            else:
                intrinsicVal = u**j*d**(steps-i-j)*S-K
                callPrices[i,j]= max(intrinsicVal, (p*callPrices[i-1,j+1] + 
                          (1-p)*callPrices[i-1,j])/(1+i))
    return callPrices
                
            
#print(binPriceCall( 0,36,0.2,40,0.06,1, 1))

def binPriceCall2(t,S, sigma, K, r, T, steps):
    C = np.zeros(shape=(steps+1,steps+1))
    
    u= np.exp(sigma*np.sqrt((T-t)/steps))
    d=1/u
    i=r**((T-t)/steps)
    p= ((1+i)-d)/(u-d)
    iteration = 0
    def intrinsicVal(i,j):
        return max(0, u**j*d**(steps-i-j)*S-K)
    
    for i in range(steps+1):
        if i==0:
            for j in range(steps+1):
                C[j, i] =  intrinsicVal(i,j)
        else:
            iteration+=1
            for j in range(steps+1-iteration):
                C[j, i] =  max(intrinsicVal(i,j), p*C[j+1,i-1]+(1-p)*C[j,i])
    return C[0,steps]

print(binPriceCall2( 0,36,0.2,40,0.06,1, 3))
