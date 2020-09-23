# Convergence of binomial model.
import BinoHull
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

Nosteps = np.arange(start=1, stop=51)
val = [BinoHull.findPayoff(steps=steps, maturity=1, spot=36, vol=0.2, strike=40, r=0.06)['0.0'] for steps in Nosteps]

rcParams['figure.figsize']=6,4
plt.style.use('ggplot')
plt.grid(True, color='k', linestyle=':') # make black grid and linestyle
plt.scatter(Nosteps, val, color='c', zorder=1)
plt.plot(Nosteps, val, color='darkgreen', zorder=2)
plt.xlabel("No. of steps")
plt.ylabel("Option Value")
plt.title('American Put: T=1, K=40, Spot=40, Vol.=0.2 and r=0.06')
plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/latex/Figures/binConv.png")
plt.show()


