
from longstaff_schwartz.algorithm import longstaff_schwartz
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion
import numpy as np
import datetime
import torch
import torch.nn as nn
from torchsummary import summary
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

# Approximation of continuation value
def fit_quadratic(x, y):
        return np.polynomial.Polynomial.fit(x, y, 2, rcond=None)

def fit_neural(x, y):
    # convert to tensors
    X = torch.from_numpy(x)
    X = X.view(len(x),1)   
    Y = torch.from_numpy(y)
    Y = Y.view(len(x),1)
    #hyperparameters
    inputSize = 1
    hidden_size1=120
    hidden_size2=120
    outputSize = 1
    learning_rate = 0.01
    
    #Design model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(NeuralNet, self).__init__()
            self.input_size = input_size
            self.l1 = nn.Linear(input_size, hidden_size1)
            self.leaky_relu_1 = nn.LeakyReLU()
            self.l2 = nn.Linear(hidden_size1,hidden_size2)
            self.leaky_relu_2 = nn.LeakyReLU()
            self.l3 = nn.Linear(hidden_size2,output_size)
        
        def forward(self,x):
            out = self.l1(x)
            out = self.leaky_relu_1(out)
            out = self.l2(out)
            out = self.leaky_relu_2(out)
            out = self.l3(out)
            return out

    model = NeuralNet(inputSize, hidden_size1, hidden_size2, outputSize)
    #model = nn.Linear(inputSize, outputSize)

    #loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    num_epochs = 1 
    #enumereate epoch
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(10):
            optimizer.zero_grad() # zero the gradient buffer
            #forward pass and loss
            y_predicted = model(X[i].float())
            loss = criterion(y_predicted,Y[i].float())
            
            # Backward and optimize
            loss.backward()
            optimizer.step() #does weight update

            # accumulate loss
            epoch_loss += loss

        epoch_loss /= len(x)
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    torch.save(model.state_dict(), "/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/python/longstaff_schwartz-master/Models/NNFit.pth")
    pass


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
                                        fit=fit_neural, exercise_payoff=put_payoff, itm_select=itm)


# Check results
print(npv_american)

finish = datetime.datetime.now()
print (finish-start)

# European put option for comparison
npv_european = constant_rate_df(t[0], t[-1]) * put_payoff(x[-1]).mean()
