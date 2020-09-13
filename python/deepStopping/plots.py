import numpy as np
from numpy.random import RandomState
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.stats.distributions import lognorm, rv_frozen
from pathlib import Path
############################
# Class to simulate paths
############################
class GeometricBrownianMotion:
    '''Geometric Brownian Motion.(with optional drift).'''
    def __init__(self, mu: float=0.0, sigma: float=1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) \
            -> np.array:
        assert t.ndim == 1, 'One dimensional time vector required'
        assert t.size > 0, 'At least one time point is required'
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), 'Increasing time vector required'
        # transposed simulation for automatic broadcasting
        dW = (rnd.normal(size=(t.size, n)).T * np.sqrt(dt)).T
        W = np.cumsum(dW, axis=0)
        return np.exp(self.sigma * W.T + (self.mu - self.sigma**2 / 2) * t).T

    def distribution(self, t: float) -> rv_frozen:
        mu_t = (self.mu - self.sigma**2/2) * t
        sigma_t = self.sigma * np.sqrt(t)
        return lognorm(scale=np.exp(mu_t), s=sigma_t)
    

S0 = 36
sigma = 0.2
# zero interest rate so that we can ignore discounting
gbm = GeometricBrownianMotion(mu=0.06, sigma=sigma)
t = np.linspace(0, 5, 12*5)
rnd = RandomState(seed=1234)
X = S0 * gbm.simulate(t, 50, rnd)
X.shape

figsize = (6, 4) # global figsize
strike = 40

#############
# Plot fitted polynomial
# cashflows and approximated continuation value
#############
cashflow = np.maximum(strike - X[-1, :], 0)
p = Polynomial.fit(X[-2, :], cashflow, 2)
p

plt.figure(figsize=figsize)
plt.grid(True, color='k', linestyle=':') # make black grid and linestyle
plt.style.use('ggplot')
plt.plot(X[-2, :], cashflow, 'g^', zorder=3);
plt.plot(*p.linspace(), color= "blue", zorder=1);
plt.plot(X[-2, :], p(X[-2, :]), 'r+', zorder=2);
plt.legend(['Cashflow',
            'Fitted Polynomial',
            'Approximated Continuation Value'])
plt.xlabel('Stock Price At Time t-1')
plt.ylabel('Time t Exercise/Continuation Value')
plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/latex/Figures/LSMFit1.png")
plt.show()

#####################
## functions to get exercise decisions
###################
intermediate_results = []

# given no prior exercise we just receive the payoff of a European option
cashflow = np.maximum(strike - X[-1, :], 0.0)
# iterating backwards in time 
for i in reversed(range(1, X.shape[0] - 1)):
    x = X[i, :]
    # exercise value for time t[i]
    exercise = np.maximum(strike - x, 0.0)
    # boolean index of all in-the-money paths
    itm = exercise > 0.0
    # fit polynomial of degree 2
    fitted = Polynomial.fit(x[itm], cashflow[itm], 2)
    # approximate continuation value
    continuation = fitted(x)
    # boolean index where exercise is beneficial
    ex_idx = itm & (exercise > continuation)
    # update cashflows with early exercises
    cashflow[ex_idx] = exercise[ex_idx]
    
    intermediate_results.append((cashflow.copy(), x, fitted, continuation, exercise, ex_idx))

def running_min_max(*array_seq):
    minimum, maximum = None, None
    for a in array_seq:
        cmin, cmax = a.min(), a.max()
        if minimum is None or cmin < minimum:
            minimum = cmin
        if maximum is None or cmax < maximum:
            maximum = cmax
    return minimum, maximum

grey = '#dddddd'

def plot_approx_n(n_steps, ax):
    cashflow, x, fitted, continuation, exercise, ex_idx = intermediate_results[n_steps]
    fitted_x, fitted_y = fitted.linspace()
    y_min, y_max = running_min_max(cashflow, exercise, fitted_y)
    offset = 0.1 * (y_max - y_min)
    ax.set_ylim((y_min - offset, y_max + offset))
    ax.plot(x, cashflow, '^', color='green', zorder=3);
    ax.plot(x[ex_idx], exercise[ex_idx], 'x', color='red', zorder=5);
    ax.plot(x[~ex_idx], exercise[~ex_idx], 'x', color=grey, zorder=4);
    ax.plot(fitted_x, fitted_y, zorder=2);
    _x = np.linspace(np.min(x), np.max(x))
    ax.plot(_x, fitted(_x), '--', color=grey, zorder=1);
    ax.legend(['Cashflow',
               'Favourable Exercise',
               'Unfavourable Exercise',
               'Approx. of Continuation Value',
               'Out-of-the-money Continuation Value'])

####################
# Exercise times
###################        
exercise_times = []
exercises = []
non_exercise_times = []
non_exercises = []
for i, (cashflow, x, fitted, continuation, exercise, ex_idx) in enumerate(intermediate_results):
    for ex in x[ex_idx]:
        exercise_times.append(t[-i-1])
        exercises.append(ex)
    for ex in x[~ex_idx]:
        non_exercise_times.append(t[-i-1])
        non_exercises.append(ex)

    
#########################
# Exercise Decision
#########################
n_timesteps, n_paths = X.shape
first_exercise_idx = n_timesteps * np.ones(shape=(n_paths,), dtype='int')
for i, (cashflow, x, fitted, continuation, exercise, ex_idx) in enumerate(intermediate_results):
    for ex in x[ex_idx]:
        idx_now = (n_timesteps - i - 1) * np.ones(shape=(n_paths,), dtype='int')
        first_exercise_idx[ex_idx] = idx_now[ex_idx]

plt.figure(figsize=figsize)
plt.style.use('ggplot')
for i in range(n_paths):
    handle_path, = plt.plot(t[0:first_exercise_idx[i]+1], X[0:first_exercise_idx[i]+1, i], '-', color='olivedrab');
    handle_stopped_path, = plt.plot(t[first_exercise_idx[i]:], X[first_exercise_idx[i]:, i], '--', color=grey);
    if first_exercise_idx[i] < n_timesteps:
        handle_first_ex, = plt.plot(t[first_exercise_idx[i]], X[first_exercise_idx[i], i], 'c^');

plt.legend([handle_path, handle_stopped_path, handle_first_ex],
           ['Path before exercise', 'Path after exercise', 'First favourable exercise'])
plt.xlabel('Time t')
plt.ylabel('Stock Value')
plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/latex/Figures/LSMFit2.png")
plt.show()

####################
## Fit exercise boundary
################
from longstaff_schwartz.binomial import create_binomial_model, american_put_price, american_put_exercise_barrier_fitted

mdl = create_binomial_model(sigma=sigma, r=1e-14, S0=S0, T=5, n=100)
exercise_barrier = american_put_exercise_barrier_fitted(mdl, strike, 3)

plt.figure(figsize=figsize)
plt.plot(exercise_times, exercises, 'rx', zorder=2)
plt.plot(*exercise_barrier.linspace(), 'g', zorder=3)
plt.plot(t, X, color='#dddddd', zorder=1)
plt.legend(['Exercise Favourable (Simulated)',
            'Fitted Exercise Boundary (Binomial Model)',
            'Simulated Paths']);
plt.xlabel('Time t')
plt.ylabel('Stock Price')
plt.show()
    
european_cashflow = np.maximum(strike - X[-1, :], 0)
assert np.average(cashflow) >= np.average(european_cashflow)
print(np.round(np.average(cashflow), 4))
print(np.round(np.average(european_cashflow), 4))