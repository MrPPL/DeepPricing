"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           Monte Carlo simulation of stock with constant volatility and risk free rate
// Course:          Master thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
// Encoding:        utf-8
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Options, Futures, and Other Derivatives by John C. Hull 10th edition
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
"""

import numpy as np
from scipy import stats
from scipy.stats import norm

#analytical formula for european call:
def priceECall(t,s, sigma, K, r, T):
    d1 = (1/(sigma*(T-t)**(0.5))) * (np.log(s/K) + (r + sigma**2 / 2) * (T-t))
    d2 = d1 - sigma * (T-t)**0.5
    return s* norm.cdf(d1) - np.exp(-r*(T-t))*K*norm.cdf(d2)

#Put-call parity proposition 9.2 in Björk
def priceEPut(t,s, sigma, K, r, T):
    return (K*np.exp(-r*(T-t)) + priceECall(t,s, sigma, K, r, T) - s)

#print(priceEPut(0,36,0.2,40,0.06,1))
#nice it worked

#Present result nicely with pandas
import pandas as pd
from pandas import Series, DataFrame

stockValues = np.linspace(36,44,5)
sigma = np.linspace(.2,.4,2)
maturity= np.linspace(1,2,2)


dict1= {'SpotPrice': [], 'Volatility': [], 
                   'Time to maturity': [], 'Closed form European' : []}
for s in stockValues:
    for volatility in sigma:
        for T in maturity:
            dict1['SpotPrice'].append(s)
            dict1['Volatility'].append(volatility)
            dict1['Time to maturity'].append(T)
            dict1['Closed form European'].append(priceEPut(0,s,volatility,40,0.06,T))
            
print(DataFrame(dict1))

