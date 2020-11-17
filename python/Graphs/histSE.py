"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           Plot of price predictions from CRR, MLP I and LSM
// Course:          Master's thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
// Encoding:        utf-8
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Option Pricing: A Simplified Approach 
// by John C. Cox, Stephen A. Ross, and Mark Rubinstein
// Neural network regression for Bermudan option pricing 
// by Bernard Lapeyre and Jérôme Lelong
// Valuing American Options by Simulation: A Simple Least-Squares Approach
// by Francis A. Longstaff and Eduardo S. Schwartz
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
"""
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
gym=pd.DataFrame({'LSM':data[:,0], 'MLPI':data[:,1]})
np.mean(gym['LSM'])
np.mean(gym['MLPI'])
from scipy.stats import sem
sem(gym['LSM'])
sem(gym['MLPI'])



rcParams['figure.figsize']=5,4
plt.style.use('ggplot')
gym.plot.hist(bins=100, alpha=0.7)
plt.xlabel("Predicted price")
plt.title("Price predictions from LSM and MLP I")
CRRPrice = BinoHull.findPayoff(steps=1000, maturity=1, spot=36, vol=0.4, strike=40, r=0.06)['0.0'] 
plt.axvline(x=CRRPrice,color='black',linestyle='--')

plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/latex/Figures/histLSMMLPsI.pdf")
plt.show()