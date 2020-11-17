"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           Monte Carlo simulation of stock with constant volatility and risk free rate
// Course:          Master's thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
//
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Options, Futures, and Other Derivatives by John C. Hull 10th edition
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
"""
import numpy as np

# Variables for Business snapshot 21.2
spot = 50
strike = 50
r = 0.05 
vol = 0.3
T = 0.5

# Simulating logreturn
#assuming vol and r constant
n = 1000 #number of simulations
normSamp = np.random.normal(loc=0, scale=1, size=n)

#find stockprice at maturity
stockPrice = [ spot*np.exp( (r-vol**2/2) * T +  vol*x*np.sqrt(T)) for x in normSamp]
print("Stockprice mean: ", np.mean(stockPrice))

#find payoff at maturtity
payOff = list(map(lambda x: x-strike if (x > strike) else 0, stockPrice))
print("payOff Mean: ", np.mean(payOff))
print("Standard deviation: ", np.std(payOff))

#Compare with closed form analytic valuation
import closedEuro
print("closed Euro Call: ", closedEuro.priceECall(t=0,s=spot, sigma=vol, K=strike, r=r, T=T))

