from longstaff_schwartz.binomial import create_binomial_model, european_call_price, american_put_price, american_put_exercise_barrier_fitted

sigma=0.2
S0=40
strike=40
mdl = create_binomial_model(sigma=sigma, r=0.06, S0=S0, T=1, n=10)
print(european_call_price(mdl, strike))