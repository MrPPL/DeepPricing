# Convergence of binomial model.
import BinoHull
import numpy as np

from matplotlib import pyplot as plt

Nosteps = np.arange(start=1, stop=51)
val = [BinoHull.findPayoff(steps=steps, maturity=1, spot=36, vol=0.2, strike=40, r=0.06)['0.0'] for steps in Nosteps]
plt.plot(Nosteps, val)
plt.xlabel("No. of steps")
plt.ylabel("Option Value")
plt.title('American put: T=1, K=40, spot=40, volatility=0.2 and r=0.06')
plt.show()


