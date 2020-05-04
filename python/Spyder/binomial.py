#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:08:48 2020

@author: ppl
"""

import closedEuro
print(closedEuro.priceECall(t=0,s=50,sigma=0.4,K=50,r=0.1, T=5/12))

def portfolio():
    if t-T<=10**(-2):
        return max(S(T)*u**(k)*d**(T-k), 0)
    else:
        V[t] = 1/(1+r) * (q * V[t+1] + (1-q)*V[t+1])