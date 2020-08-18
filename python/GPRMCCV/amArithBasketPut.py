# Halton sequence implementation
# link https://laszukdawid.com/2017/02/04/halton-sequence-in-python/
def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True
 
    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq

######################
#######################
#parameters
K = 100
r = 0.05
spot = 100
vol = 0.2
T =1
eta = 0 #dividend yield
d = 2 # dimension
rho = 0.2 #correlation

N = 10 # number of exercise dates 
P = 250 # points
M = 10**3 # monte carlo simulations
timeInc = T/N # time increment
Q = 5 # points for computation of European prices

def arithPutPayOff(K, S):
    """K is the strike price, S is a vector of stock prices"""
    d = len(S) #dimension of S
    return max(K-sum(S)/d)

#################
## GPR - EI formula
#################
import numpy as np
corrMatrix = np.array([[1, 0.2],[0.2, 1]])

Z = np.zeros((d,Q)) 
h = halton_sequence(Q, d) #pseudo random sequence
for i in range(d):
    Z[i] = np.sqrt(T)*vol*np.matmul(np.sqrt(corrMatrix[i]),h)
print(Z)

def u(spot, z, r, eta, vol, payoff):
    val = spot*np.exp(r-eta-vol^2/2)*T+z
    return payoff(val)
