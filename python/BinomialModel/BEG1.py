#packages
import numpy as np
#Parameter
S = np.array([100,100])
vol = np.array([0.2,0.2])
corr = 0.5
r = 0.1
T = 1
K = 100
n = 2 #dimensions
N = 3 #number of steps T is divided into to
h = T/N

# jumpSizes
u = np.exp(vol*np.sqrt(h))
d = 1/u


#make tree
tree = {}
for timestep in range(N+1):
    if timestep==0:
        tree[timestep] = S
    else:
        list1 = []
        for item in tree[timestep-1]:
            for i in range(1):
                if i==0:
                    list1.append(item*d)
