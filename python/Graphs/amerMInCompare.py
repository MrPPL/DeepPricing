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
plt.scatter(df.Spot, df.MLPsII, label='MLPsII', marker='+', color='green')
plt.legend()
plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/latex/Figures/compareBEGMLPsII.pdf")
plt.show() 




