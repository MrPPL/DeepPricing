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

# Approximation of continuation value
#hyperparameters
n_samples, n_features = len(dataset), len(features)
input_size = n_features
hidden_size1 = 12
outputSize = 1
num_epochs = 10
batchSize = 64
learning_rate = 0.01

#Design model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.leaky_relu_1 = nn.LeakyReLU()
        self.l2 = nn.Liner(hidden_size1,output_size)
    
    def forward(self,x):
        out = self.l1(x)
        out = self.leaky_relu_1(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size1, outputSize)

#loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
#enumereate epoch
print(optimizer.state_dict())
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (X, y) in enumerate(train_loader):  #one batch of samples       
        optimizer.zero_grad() # zero the gradient buffer

        #forward pass and loss
        y_predicted = model(X)
        loss = criterion(y_predicted,y)
        
        # Backward and optimize
        loss.backward()
        optimizer.step() #does weight update

        # accumulate loss
        epoch_loss += loss

    epoch_loss /= n_total_steps
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print(optimizer.state_dict())


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
                                        fit=neuralNetFit, exercise_payoff=put_payoff, itm_select=itm)


# Check results
print(npv_american)

finish = datetime.datetime.now()
print (finish-start)

# European put option for comparison
npv_european = constant_rate_df(t[0], t[-1]) * put_payoff(x[-1]).mean()