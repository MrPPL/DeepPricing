"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           Plot of price predictions from BEG and MLPs II
// Course:          Master's thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
// Encoding:        utf-8
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Numerical Evaluation of Multivariate Contingent Claims 
// by Phelim P. Boyle, Jeremy Evnine, and Stephen Gibbs
// Supervised Deep Neural Networks (DNNs) for Pricing/Calibration of 
// Vanilla/Exotic Options Under Various Different Processes
// by Tugce Karatas, Amir Oskoui, and Ali Hirsa
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

df = pd.read_csv (r'./../AmericanPutMinCompare.csv')
print (df)
df.head()
sum(abs(df.BEG500-df.MLPsII))
sum(abs(df.BEG50-df.MLPsII))
sum(abs(df.BEG500-df.BEG50))

sum(abs(df.BEG500-df.MLPsII))/21

# scatter plot 
rcParams['figure.figsize']=5,4
plt.style.use('ggplot')
plt.xlabel("Spot for both stocks")
plt.ylabel("Predicted price")
plt.title("Price predictions from BEG and MLP II")
plt.scatter(df.Spot, df.BEG500, label='BEG500', marker='^', color='blue')
plt.scatter(df.Spot, df.BEG50, label='BEG50', marker='*', color='red')
plt.scatter(df.Spot, df.MLPsII, label='MLPII', marker='+', color='green')
plt.legend()
plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/latex/Figures/compareBEGMLPsII.pdf")
plt.show() 




