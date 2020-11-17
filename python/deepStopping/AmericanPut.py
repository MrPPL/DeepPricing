"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           LSM and MLP I pricing methods
// Course:          Master's thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
// Encoding:        utf-8
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
// Neural network regression for Bermudan option pricing 
// by Bernard Lapeyre and Jérôme Lelong
// Valuing American Options by Simulation: A Simple Least-Squares Approach
// by Francis A. Longstaff and Eduardo S. Schwartz
// Build on Luphord's Github: https://github.com/luphord/longstaff_schwartz
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
"""


from longstaff_schwartz.algorithm import longstaff_schwartz
import sys 
import os
sys.path.append(os.path.abspath("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepStopping/longstaff_schwartz"))
import algorithm1
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion
import numpy as np
import datetime
import torch
import torch.nn as nn
from torchsummary import summary

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
#read file
import sys 
import os
sys.path.append(os.path.abspath("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepLearning/minAmerican"))
import earlyStop


start = datetime.datetime.now()

global count10 #load model or not
count10 = 0 
# Model parameters
t = np.linspace(0, 2, 100)  # timegrid for simulation
r = 0.06  # riskless rate
sigma = 0.4  # annual volatility of underlying
n = 10**5  # number of simulated paths

# Simulate the underlying
gbm = GeometricBrownianMotion(mu=r, sigma=sigma)
rnd = np.random.RandomState(100)
x = gbm.simulate(t, n, rnd)  # x.shape == (t.size, n)

# Payoff (exercise) function
strike = 40

def put_payoff(spot):
        return np.maximum(strike - spot, 0.0)

# Discount factor function
def constant_rate_df(t_from, t_to):
        return np.exp(-r * (t_to - t_from))

# Approximation of continuation value
def fit_quadratic(x, y):
        return np.polynomial.Polynomial.fit(x, y, 10, rcond=None)

def fit_neural(x, y):
        # convert to tensors
        X = torch.Tensor(x)
        X = X.view(len(x),1)   
        Y = torch.Tensor(y)
        Y = Y.view(len(x),1)

        my_dataset = TensorDataset(X,Y) # create your datset

        batchSize = 512
        train_loader = DataLoader(dataset=my_dataset,
                                batch_size=batchSize,
                                num_workers=2,
                                shuffle=True)
        #hyperparameters
        inputSize = 1
        hidden_size1=40
        hidden_size2=40
        outputSize = 1
        learning_rate = 0.001

        #Design model
        class NeuralNet(nn.Module):
                def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
                        super(NeuralNet, self).__init__()
                        self.input_size = input_size
                        self.l1 = nn.Linear(input_size, hidden_size1)
                        self.leaky_relu_1 = nn.LeakyReLU(negative_slope=0.3)
                        self.l2 = nn.Linear(hidden_size1,hidden_size2)
                        self.leaky_relu_2 = nn.LeakyReLU(negative_slope=0.3)
                        self.l3 = nn.Linear(hidden_size2,outputSize)
                
                def forward(self,x):
                        out = self.l1(x)
                        out = self.leaky_relu_1(out)
                        out = self.l2(out)
                        out = self.leaky_relu_2(out)
                        out = self.l3(out)
                        return out
        global count10
        if count10 == 0:
                model = NeuralNet(inputSize, hidden_size1,hidden_size2, outputSize)
                count10 +=1
        else: 
                model = NeuralNet(inputSize, hidden_size1, hidden_size2, outputSize)
                model.load_state_dict(torch.load("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepStopping/saveModel/ModelAM1.pth"))
        #model = nn.Linear(inputSize, outputSize)

        #loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        num_epochs = 1
        n_total_steps = len(train_loader)
        #enumereate epoch
        es = earlyStop.EarlyStopping(patience=1)
        for epoch in range(num_epochs):
                total_loss = 0
                n_samples = 0
                for i, (X, y) in enumerate(train_loader):  #one batch of samples       
                        optimizer.zero_grad() # zero the gradient buffer
                        #forward pass and loss
                        y_predicted = model(X)
                        loss = criterion(y_predicted,y)
                        # Backward and optimize
                        loss.backward()
                        optimizer.step() #does weight update
                        #epoch_loss += loss
                        # accumulate loss
                        total_loss += loss.item() * X.shape[0]
                        n_samples += X.shape[0]
                total_loss /= n_samples
                #if (epoch+1) % 10 == 0: 
                #print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
                if es.step(total_loss):
                    break  # early stop criterion is met, we can stop now
        #enumereate epoch
        torch.save(model.state_dict(), "/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepStopping/saveModel/ModelAM1.pth")
        return model


# Selection of paths to consider for exercise
# (and continuation value approxmation)
def itm(payoff, spot):
        return payoff > 0

# Run valuation of American put option
# X=simulated path
# t=timegrad for simulation
# df = discounting for a periode
# fit = fit_quadratic
#for spot in range(36,46,2):
#        npv_american = longstaff_schwartz(X=x*spot, t=t, df=constant_rate_df,
#                                        fit=fit_neural, exercise_payoff=put_payoff, itm_select=itm)
#        npv1_american = algorithm1.longstaff_schwartz(X=x*spot, t=t, df=constant_rate_df,
#                                        fit=fit_quadratic, exercise_payoff=put_payoff, itm_select=itm)
#
#        # Print results
#        print("spot", spot, "vol", sigma, "MLPsI", npv_american)
#        print("spot", spot, "vol", sigma,"LSM", npv1_american)


# European put option for comparison
npv_european = constant_rate_df(t[0], t[-1]) * put_payoff(x[-1]).mean()

import pandas as pd
def findY(size):
        Y = np.empty((size,1))
        Z = np.empty((size,1))
        global count10 #load model or not
        # Model parameters
        t = np.linspace(0, 1, 50)  # timegrid for simulation
        r = 0.06  # riskless rate
        sigma = 0.4  # annual volatility of underlying
        n = 10**5  # number of simulated paths
        # Payoff (exercise) function
        strike = 40
        spot=36
        gbm = GeometricBrownianMotion(mu=r, sigma=sigma)
        rnd = np.random.RandomState(100)
        for i in range(size):
                # Simulate the underlying
                x = gbm.simulate(t, n, rnd)  # x.shape == (t.size, n)
                count10 = 0 
                #MLPs I
                Y[i] = longstaff_schwartz(X=x*spot, t=t, df=constant_rate_df,
                                        fit=fit_neural, exercise_payoff=put_payoff, itm_select=itm)
                #LSM
                Z[i] = algorithm1.longstaff_schwartz(X=x*spot, t=t, df=constant_rate_df,
                                        fit=fit_quadratic, exercise_payoff=put_payoff, itm_select=itm)
                                        
                print("spot", spot, "vol", sigma, "MLPsI", Y[i])
        
        df = pd.DataFrame({'LSM':Z[:,0],'MLPs I':Y[:,0]})
        return df

df = findY(100)
df.to_csv("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepStopping/SEData/36Spot4vol1Y.csv")