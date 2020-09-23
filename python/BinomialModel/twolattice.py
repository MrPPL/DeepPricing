##########
## Tree method
##########
#model parameters
S10 = 100
S20 = 100
K=200
Sb=260
r=0.01
sigma1 = 0.3
sigma2 = 0.2
T = 1
rho = 0.5
NStep = 10
import numpy as np

def BEG(Nstep, T, sigma1, sigma2, r, rho, S10, S20, K):
    # invariant quantities
    N = NStep
    deltaT = T/N
    nu1 = r - 0.5*sigma1**2
    nu2 = r - 0.5*sigma2**2
    u1 = np.exp(sigma1*np.sqrt(deltaT))
    d1 = 1/u1
    u2 = np.exp(sigma2*np.sqrt(deltaT))
    d2 = 1/u2
    discount = np.exp(-r*deltaT)
    p_uu = discount*0.25*(1+np.sqrt(deltaT)*(nu1/sigma1+nu2/sigma2)+rho)
    p_ud = discount*0.25*(1+np.sqrt(deltaT)*(nu1/sigma1-nu2/sigma2)-rho)
    p_du = discount*0.25*(1+np.sqrt(deltaT)*(-nu1/sigma1+nu2/sigma2)-rho)
    p_dd = discount*0.25*(1+np.sqrt(deltaT)*(-nu1/sigma1-nu2/sigma2)+rho)
    
    # set up stock values
    S1Vals = np.zeros((2*N+1,1))
    S2Vals = np.zeros((2*N+1,1))

    S1Vals[0] = S10*d1**N
    S2Vals[0] = S20*d2**N
    for i in range(1,2*N+1):
        S1Vals[i] = u1*S1Vals[i-1]
        S2Vals[i] = u2*S2Vals[i-1]
    
    # set up terminal values
    CVals = np.zeros((2*N+1,2*N+1))
    
    for i in range(0,2*N+1,2):
        for j in range(0,2*N+1,2):
            if (S1Vals[i] + S2Vals[j]) >= Sb:
                CVals[i,j] = 0
            else:
                CVals[i,j] = max(0,S1Vals[i]+S2Vals[j]-K)
            
    # roll back
    for tau in range(N):
        for i in range(tau+1,2*N+1-tau,2):
            for j in range(tau+1,2*N+1-tau,2):
                if (S1Vals[i] + S2Vals[j]) >= Sb:
                    CVals[i,j] = 0
                else:
                    CVals[i,j] = p_uu*CVals[i+1,j+1]+ p_ud*CVals[i+1,j-1]+p_du*CVals[i-1,j+1]+p_dd*CVals[i-1,j-1]

    price = CVals[N,N]
    return price


#model parameters
S10 = 100
S20 = 100
K=200
Sb=260
r=0.01
sigma1 = 0.3
sigma2 = 0.2
T = 1
rho = 0.5
NStep = 1000

print(BEG(Nstep=NStep, T=T, sigma1=sigma1, sigma2=sigma2, r=r, rho=rho, S10=S10, S20=S20, K=K))