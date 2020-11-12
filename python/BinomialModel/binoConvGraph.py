# Convergence of binomial model.
import BinoHull
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

Nosteps = np.arange(start=1, stop=201)
val = [BinoHull.findPayoff(steps=steps, maturity=1, spot=36, vol=0.2, strike=40, r=0.06)['0.0'] for steps in Nosteps]

rcParams['figure.figsize']=5,4
plt.style.use('ggplot')
plt.scatter(Nosteps, val, color='c', zorder=1)
plt.plot(Nosteps, val, color='darkgreen', zorder=2)
plt.xlabel("No. of steps")
plt.ylabel("Option Value")
plt.title('Convergence American Put') 
#plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/latex/Figures/binConv.pdf")
plt.show()


