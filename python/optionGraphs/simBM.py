#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:32:53 2020

@author: ppl
"""
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(123)
np.random.seed(123) # set the seed for numpy



def weinerProcess(steps, time):
    timePoint = 0 # initial time
    position = (timePoint,0) # starting point for wiener process
    path = [position]
    t = time # time horisont
    n = steps # number of steps
    incr = t/n #equal increment size
    
    for i in range(1,n+1):
        tmpTime=timePoint
        timePoint+=incr
        timeStep = timePoint - tmpTime
        wInc = np.sqrt(timeStep)*np.random.normal(0, 1, size=1)[0]
        w = wInc + position[1]
        position = (timePoint, w)
        path.append(position)
    
    return path
        
y = weinerProcess(1000,2)

x_val = [x[0] for x in y]
y_val = [x[1] for x in y]

plt.plot(x_val,y_val)
plt.xlabel("Time t")
plt.ylabel("W(t)")
plt.title("A Brownian Motion trajectory")
plt.show()

#
#plt.plot(seq, funcDict["exp(x**2)"], label="exp(x**2)")
#plt.plot(seq, funcDict["exp(2x)"], label="exp(2*x)")
#plt.xlabel('x label')
#plt.ylabel('y label')
#plt.title("Compare growth")
#plt.legend()
plt.show()

