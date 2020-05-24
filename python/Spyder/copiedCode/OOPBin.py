#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:50:06 2020

@author: ppl
"""

## copied code from http://www.josephthurman.com/binomial4.html
import math

class Stock:
    def __init__(self, spot, vol):
        self.spot = spot
        self.vol = vol
        
class Option:
    def __init__(self, underlying, expiry):
        self.underlying = underlying
        self.expiry = expiry

    def final_payoff(self, spot):
        raise NotImplementedError("Final option payoff is not defined")

    def early_payoff(self, spot):
        raise NotImplementedError("Early exercise payoff is not defined")
        
        
class EuroCall(Option):
    def __init__(self, underlying, expiry, strike):
        super().__init__(underlying, expiry)
        self.strike = strike

    def final_payoff(self, spot):
        return max(spot - self.strike,0)

    def early_payoff(self, spot):
        return 0

class EuroPut(Option):
    def __init__(self, underlying, expiry, strike):
        super().__init__(underlying, expiry)
        self.strike = strike

    def final_payoff(self, spot):
        return max(self.strike - spot,0)

    def early_payoff(self, spot):
        return 0
    
class AmerCall(EuroCall):
    def early_payoff(self, spot):
        return self.final_payoff(spot)

class AmerPut(EuroPut):
    def early_payoff(self, spot):
        return self.final_payoff(spot)
    
class BinomialModel:
    def __init__(self, option, r):
        self.option = option
        self.r = r

    def price(self, N=500):
        dt = self.option.expiry/N
        u =  math.exp(self.option.underlying.vol * math.sqrt(dt))
        d = 1/u
        p = (math.exp(self.r * dt) - d)/(u - d)

        # Computes the price of the underlying asset k steps into the tree with m up movements
        def S(k,m):
            return self.option.underlying.spot * (u ** (2*m-k))

        C = {}
        for m in range(0, N+1):
            C[(N, m)] = self.option.final_payoff(S(N,m))
        for k in range(N-1, -1, -1):
            for m in range(0,k+1):
                future_value = math.exp(-self.r * dt) * (p * C[(k+1, m+1)] + (1-p) * C[(k+1, m)])
                exercise_value = self.option.early_payoff(S(k,m))
                C[(k, m)] = max(future_value, exercise_value)
        return C[(0,0)]

test_stock = Stock(100, 0.2)
euro = EuroCall(test_stock,0.25,115)
print("EuroC", BinomialModel(euro,0.015).price())

amer = AmerCall(test_stock,0.25,115)
print("AmerC", BinomialModel(amer,0.015).price())

amerp = AmerPut(test_stock,0.25,115)
print("EuroP", BinomialModel(amerp,0.015).price())
europ = EuroPut(test_stock,0.25,115)
print("AmerP", BinomialModel(europ,0.015).price())
