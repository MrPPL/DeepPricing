#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:51:57 2020

@author: ppl
"""

import numpy as np

#d= 1/u
#u= np.exp(sigma*np.sqrt(t/n))
#rHat = r**(t/n)
import math

def binomial_call(S, K, T, r, vol, N):
    """
    Implements the binomial option pricing model to price a European call option on a stock
    S - stock price today
    K - strike price of the option
    T - time until expiry of the option
    r - risk-free interest rate
    vol - the volatility of the stock
    N - number of steps in the model
    """
    dt = T/N
    u =  math.exp(vol * math.sqrt(dt))
    d = 1/u
    p = (math.exp(r * dt) - d)/(u - d)
    C = {}
    for m in range(0, N+1):
            C[(N, m)] = max(S * (u ** (2*m - N)) - K, 0)
    for k in range(N-1, -1, -1):
        for m in range(0,k+1):
            C[(k, m)] = math.exp(-r * dt) * (p * C[(k+1, m+1)] + (1-p) * C[(k+1, m)])
    return C[(0,0)]

import numpy as np
import closedEuro
print(closedEuro.priceECall(t=0,s=50,sigma=0.4,K=50,r=0.1, T=5/12))

t,S, sigma, K, r, T = 0,36,0.2,50,0.1,5712

N = 100
print(binomial_call(S=50,K=50, T=5/12, r=0.1,  vol=0.4, N=999))

#check

vol=0.2; K=35; N=5; T=7; S=40;r=0.05;
print(binomial_call(S, K, T, r, vol, N))
