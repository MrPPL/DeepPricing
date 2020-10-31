import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
#read file
import sys 
import os
sys.path.append(os.path.abspath("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/BinomialModel"))
import BinoHull
data = np.loadtxt("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepStopping/SEData/36Spot4vol1Y.csv", delimiter=',', dtype=np.float32, skiprows=1)
data=data[:,1:]
gym=pd.DataFrame({'LSM':data[:,0], 'MLPsI':data[:,1]})
np.mean(gym['LSM'])
np.mean(gym['MLPsI'])
from scipy.stats import sem
sem(gym['LSM'])
sem(gym['MLPsI'])



rcParams['figure.figsize']=5,4
plt.style.use('ggplot')
gym.plot.hist(bins=100, alpha=0.7)
plt.xlabel("Predicted Price")
plt.title("Price Predictios from LSM and MLPs I")
CRRPrice = BinoHull.findPayoff(steps=1000, maturity=1, spot=36, vol=0.4, strike=40, r=0.06)['0.0'] 
plt.axvline(x=CRRPrice,color='black',linestyle='--')

plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/latex/Figures/histLSMMLPsI.png")
plt.show()