from longstaff_schwartz.algorithm import longstaff_schwartz
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion
import numpy as np
import datetime
start = datetime.datetime.now()

# Model parameters
t = np.linspace(0, 1, 100)  # timegrid for simulation
r = 0.06  # riskless rate
sigma = 0.2  # annual volatility of underlying
n = 10**5  # number of simulated paths

# Simulate the underlying
gbm = GeometricBrownianMotion(mu=r, sigma=sigma)
rnd = np.random.RandomState(1234)
x = 36 * gbm.simulate(t, n, rnd)  # x.shape == (t.size, n)

# Payoff (exercise) function
strike = 40

def put_payoff(spot):
        return np.maximum(strike - spot, 0.0)

# Discount factor function
def constant_rate_df(t_from, t_to):
        return np.exp(-r * (t_to - t_from))

def fit_quadratic(x, y):
        return np.polynomial.Polynomial.fit(x, y, 2, rcond=None)

# Selection of paths to consider for exercise
# (and continuation value approxmation)
def itm(payoff, spot):
        return payoff > 0

# Run valuation of American put option
# X=simulated path
# t=timegrad for simulation
# df = discounting for a periode
# fit = fit_quadratic
npv_american = longstaff_schwartz(X=x, t=t, df=constant_rate_df,
                                        fit=fit_quadratic, exercise_payoff=put_payoff, itm_select=itm)


# Check results
print(npv_american)

finish = datetime.datetime.now()
print (finish-start)

# European put option for comparison
npv_european = constant_rate_df(t[0], t[-1]) * put_payoff(x[-1]).mean()