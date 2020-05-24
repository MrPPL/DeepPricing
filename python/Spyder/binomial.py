#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:08:48 2020

@author: ppl
"""
import numpy as np
import closedEuro
print(closedEuro.priceECall(t=0,s=50,sigma=0.4,K=50,r=0.1, T=5/12))

t,S, sigma, K, r, T = 0,36,0.2,40,0.06,1

steps = 5



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
                
            
print(binPriceCall( 0,36,0.2,40,0.06,1, 1))

#def portfolio():
#    if t-T<=10**(-2):
#        return max(S(T)*u**(k)*d**(T-k), 0)
#    else:
#        V[t] = 1/(1+r) * (q * V[t+1] + (1-q)*V[t+1])