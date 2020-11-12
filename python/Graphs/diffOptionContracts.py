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

plt.style.use('ggplot')
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(6, 4)
x = np.arange(0, 200, 0.5)
y = [EuroCall(100, S) for S in x]
ax[0].set_title('Call With Strike K=100')
ax[0].set_xlabel("S(T)")
ax[0].set_ylabel("Payoff")
ax[0].plot(x, y, 'r', linewidth=1)
x1 = np.arange(0, 200, 0.5)
y1 = [EuroPut(100, S) for S in x]
ax[1].set_title('Put With Strike K=100')
ax[1].set_xlabel("S(T)")
ax[1].plot(x1, y1, 'c', linewidth=1)
#plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/latex/Figures/contractfct.pdf")
plt.show()
