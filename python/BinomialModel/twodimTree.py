import numpy as np
N=3 # number of steps
vol = [0.2,0.2] #volatility
rho = 0.5 # correlation
spot = [100,100]
T = 1
r=0.1
K=100

def tree(N, T, r, vol, rho):
    # invariant quantities
    deltaT = T/N
    mu1 = r - 0.5*vol[0]**2
    mu2 = r - 0.5*vol[1]**2
    u1 = np.exp(vol[0]*np.sqrt(deltaT))
    d1 = 1/u1
    u2 = np.exp(vol[1]*np.sqrt(deltaT))
    d2 = 1/u2
    discount = np.exp(r*deltaT)
    p_uu = discount*0.25*(1+np.sqrt(deltaT)*(mu1/vol[0]+mu2/vol[1])+rho)
    p_ud = discount*0.25*(1+np.sqrt(deltaT)*(mu1/vol[0]-mu2/vol[1])-rho)
    p_du = discount*0.25*(1+np.sqrt(deltaT)*(-mu1/vol[0]+mu2/vol[1])-rho)
    p_dd = discount*0.25*(1+np.sqrt(deltaT)*(-mu1/vol[0]-mu2/vol[1])+rho)
    
    # set up stock values
    S1Vals = np.zeros((2*N+1,1))
    S2Vals = np.zeros((2*N+1,1))
    S1Vals[0] = spot[0]*d1**N
    S2Vals[0] = spot[1]*d2**N
    for i in range(1,2*N):
        S1Vals[i] = u1*S1Vals[i-1]
        S2Vals[i] = u2*S2Vals[i-1]
    
    # set up terminal values
    CVals = np.zeros((2*N+1,2*N+1))
    for i in range(0,2*N, 2):
        for j in range(0,2*N, 2):
            CVals[i,j] = max(max(S1Vals[i], S2Vals[j])-K,0)

    # roll back
    for tau in range(N):
        for i in range(tau,2*N-tau,2):
            for j in range(tau,2*N-tau,2):
                CVals[i,j] = p_uu*CVals[i+1,j+1]+ p_ud*CVals[i+1,j-1] + p_du*CVals[i-1,j+1]+p_dd*CVals[i-1,j-1]

    price = CVals[N+1,N+1]
    return (price)


print(tree(N, T, r, vol, rho))