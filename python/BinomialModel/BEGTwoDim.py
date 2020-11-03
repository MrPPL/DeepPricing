##########
## Tree method 2-dimensionel inspired by BEG
##########
import numpy as np
import datetime
def BEG(Nstep, T, sigma1, sigma2, r, rho, S10, S20, K, amer):
    # measure time
    start = datetime.datetime.now()
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
            CVals[i,j] = max(0,K-min(S1Vals[i],S2Vals[j]))
            
    # roll back
    for tau in range(N):
        for i in range(tau+1,2*N+1-tau,2):
            for j in range(tau+1,2*N+1-tau,2):
                if amer==True:
                    CVals[i,j] = max(max(0,K-min(S1Vals[i],S2Vals[j])) ,p_uu*CVals[i+1,j+1]+ p_ud*CVals[i+1,j-1]+p_du*CVals[i-1,j+1]+p_dd*CVals[i-1,j-1])
                else:
                    CVals[i,j] = p_uu*CVals[i+1,j+1]+ p_ud*CVals[i+1,j-1]+p_du*CVals[i-1,j+1]+p_dd*CVals[i-1,j-1]
                    
    price = CVals[N,N]
    finish = datetime.datetime.now()
    print ("Time taken",finish-start)
    print(price)
    return price


#model parameters
S10 = 40
S20 = 40
K= 40
r=0.06
sigma1 = 0.2
sigma2 = 0.3
T = 1
rho = 0.5
NStep = 500
for spot in range(30,51,1):
    print("spot", spot, "BEG", BEG(Nstep=NStep, T=T, sigma1=sigma1, sigma2=sigma2, r=r, rho=rho, S10=spot, S20=spot, K=K, amer=True))

#First order homogeneous function
#print(BEG(Nstep=50, T=1, sigma1=0.2, sigma2=0.3, r=0.06, rho=-0.5, S10=120, S20=10, K=40, amer=True))
#a = BEG(Nstep=50, T=1, sigma1=0.2, sigma2=0.3, r=0.06, rho=-0.5, S10=120/40, S20=10/40, K=40/40, amer=True)
#print(a*40)
#print(BEG(Nstep=50, T=0.003968, sigma1=0.05, sigma2=0.05, r=0.01, rho=-0.5, S10=0.8, S20=0.8, K=1, amer=True))