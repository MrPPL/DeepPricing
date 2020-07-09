#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:57:02 2020

@author: ppl
"""
import numpy as np


"""This file is basically to graph all the option pricing contracts"""

def EuroCall(K,S):
    return ( max(S-K, 0))

def EuroPut(K,S):
    return ( max(K-S, 0))


from matplotlib import pyplot as plt

x = np.arange(0, 200, 0.5)
y = [EuroCall(100, S) for S in x]
plt.plot(x,y)
plt.xlabel("Underlying asset: S(T)")
plt.ylabel("Value of derivative X")
plt.title('European Call option with strike 100')
plt.show()

x = np.arange(0, 200, 0.5)
y = [EuroPut(100, S) for S in x]
plt.plot(x,y)
plt.xlabel("Underlying asset: S(T)")
plt.ylabel("Value of derivative X")
plt.title('European Put option with strike 100')
plt.show()