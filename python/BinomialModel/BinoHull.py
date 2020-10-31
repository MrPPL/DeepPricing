#Binomial tree in Hull p 453 
#packages
import numpy as np
import datetime

## Example 21.1
# American put option 5 month
#spot = 40
#strike = 40
#r=0.06
#vol = 0.2
#maturity = 1
#steps = 50


#Make tree dictionary
def binStockPath(stepSize, steps, spot, vol):
    STree = {}
    #stepSize = maturity/steps
    u = np.exp(vol*stepSize**0.5)
    d = np.exp(-vol*stepSize**0.5)
    for i in range(steps+1):
        if i==0:
            STree["{}".format(i*stepSize)] = spot
        else:
            list1 = []
            [list1.append(spot*u**j*d**(i-j)) for j in range(i+1)]

            STree["{}".format(i*stepSize)] = list1
    return STree

#intrinsic value dictionary for american put option
def findIntrinsicTree(stepSize, steps, spot, vol, strike):
    #stepSize = maturity/steps
    STree = binStockPath(stepSize, steps, spot, vol)
    intrinsicTree = {}
    for i in reversed(range(steps+1)):
        if i==0:
            intrinsicTree["{}".format(i*stepSize)] = max(strike-STree["{}".format(i*stepSize)],0) 
        else:
            intrinsicTree["{}".format(i*stepSize)] = [max(strike-item,0) for item in STree["{}".format(i*stepSize)]]
    return intrinsicTree


#find condition expection
def findPayoff(steps, maturity, spot, vol, strike, r):
    start = datetime.datetime.now()
    stepSize = maturity/steps
    intrinsicTree = findIntrinsicTree(stepSize, steps, spot, vol, strike)
    u = np.exp(vol*stepSize**0.5)
    d = np.exp(-vol*stepSize**0.5)
    a = np.exp(r*stepSize)
    # risk neutral probabilities
    p = (a-d)/(u-d)
    expecTree = {}
    for i in reversed(range(steps+1)):
        if i==steps:
            expecTree["{}".format(i*stepSize)] = intrinsicTree["{}".format(i*stepSize)]
            continue
        else:
            list1 = [None]*(i+1)
            for count in range(i+1):
                condExp = 0
                for k in range(count, count+1):
                    condExp += (1-p)*expecTree["{}".format((i+1)*stepSize)][k]
                    k+=1
                    condExp += (p) * expecTree["{}".format((i+1)*stepSize)][k]
                    condExp = condExp*np.exp(-r*stepSize) # discount back
                if (i==0):
                    list1[count] = condExp 
                elif (condExp > intrinsicTree["{}".format(i*stepSize)][count]):
                    list1[count] = condExp
                else:
                    list1[count] = intrinsicTree["{}".format(i*stepSize)][count]
        expecTree["{}".format(i*stepSize)]=list1
    finish = datetime.datetime.now()
    print (finish-start)
    return expecTree

print(findPayoff(steps=50, maturity=1, spot=36, vol=0.2, strike=40, r=0.06)['0.0'])
#a = (findPayoff(steps=100, maturity=2, spot=40/40, vol=0.2, strike=40/40, r=0.06)['0.0'])
#print(a[0]*40)

#for S in range(36,46, 2):
#    print(findPayoff(steps=200, maturity=2, spot=S, vol=0.4, strike=40, r=0.06)['0.0'])

from longstaff_schwartz.binomial import create_binomial_model, american_put_price, american_put_exercise_barrier_fitted
sigma=0.2
S0=36
strike=40
mdl = create_binomial_model(sigma=sigma, r=0.06, S0=S0, T=1, n=50)
print(american_put_price(mdl, strike))
