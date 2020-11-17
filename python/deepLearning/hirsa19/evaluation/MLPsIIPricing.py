"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           Price predictions based on trained models for European call,
//                  American put, and American put minimum on two stocks
// Course:          Master's thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
// Encoding:        utf-8
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
// Supervised Deep Neural Networks (DNNs) for Pricing/Calibration of 
// Vanilla/Exotic Options Under Various Different Processes
// by Tugce Karatas, Amir Oskoui, and Ali Hirsa
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

###########################
#Model used for training
###########################
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.leaky_relu_1 = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.leaky_relu_2 = nn.LeakyReLU()
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.leaky_relu_3 = nn.LeakyReLU()
        self.l4 = nn.Linear(hidden_size3, outputSize)
        self.leaky_relu_4 = nn.LeakyReLU()
    
    def forward(self,x):
        out = self.l1(x)
        out = self.leaky_relu_1(out)
        out = self.l2(out)
        out = self.leaky_relu_2(out)
        out = self.l3(out)
        out = self.leaky_relu_3(out)
        out= self.l4(out)
        out = self.leaky_relu_4(out)
        return out

##########
# Dataset Class
##########
class EuroParDataset(Dataset):
    def __init__(self, dataPath):
        # Initialize data, download, etc.
        xy = np.loadtxt(dataPath, delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 2:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [1]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    # get indexes for train and test rows
    def get_splits(self, n_test=0.2, n_valid=0.2):
        # determine sizes
        #test_size = round(n_test * len(self.x_data))
        valid_size = round(n_valid * len(self.x_data))
        train_size = len(self.x_data) - valid_size
        # calculate the split
        return random_split(self, [train_size, valid_size])
#prepare data
dataPath = "./deepLearning/hirsa19/data/300KCEuroData.csv"
dataset = EuroParDataset(dataPath)
# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
#define network
#hyperparameters
input_size = dataset.x_data.shape[1]
hidden_size1 = 120
hidden_size2 = 120
hidden_size3 = 120
outputSize = 1

########################
#load trained model
###################
loaded_model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, outputSize)
#for param in loaded_model.parameters():
#    print("without training model", param)
loaded_model.load_state_dict(torch.load("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepLearning/Models/hirsaModelEC2.pth"))
loaded_model.eval()
#loaded_model.state_dict()
#for param in loaded_model.parameters():
#    print("loaded model", param)

############
# Make predictions
###########
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = torch.tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

#row maturity, rate, vol, moneyness
# european call
vol = 0.2
moneyness = 1
strike = 40
r = 0.06 
T =1
row = [moneyness,T, r, vol]
yhat = predict(row, loaded_model) * strike
print('Predicted European Call: %.3f' % yhat)

##############################
# American Put
#############################
loaded_model2 = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, outputSize)
#for param in loaded_model.parameters():
#    print("without training model", param)
loaded_model2.load_state_dict(torch.load("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepLearning/Models/hirsaModelAM1.pth"))
loaded_model2.eval()
#row maturity, rate, vol, moneyness
dataPath = "./deepLearning/hirsa19/data/300KPAmerData.csv"
dataset = EuroParDataset(dataPath)
# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
# european call
vol = 0.25
moneyness = (100/110)
strike = 110
r = 0.06 
T =1
row = [moneyness,T, r, vol]
yhat = predict(row, loaded_model2) * strike
print('Predicted American Put: %.3f' % yhat)


##############################################
## American minimum put option
################################################
#row maturity, rate, vol, moneyness
dataPath = "./deepLearning/minAmerican/data1/300KAmerMinPut.csv"
dataset = EuroParDataset(dataPath)
# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
#hyperparameters
input_size = dataset.x_data.shape[1]
hidden_size1 = 120
hidden_size2 = 120
hidden_size3 = 120
outputSize = 1
loaded_model3 = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, outputSize)
#for param in loaded_model.parameters():
#    print("without training model", param)
loaded_model3.load_state_dict(torch.load("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepLearning/Models/ModelAM_Min2.pth"))
loaded_model3.eval()
# american min put

import datetime
start = datetime.datetime.now()
strike = 40
spot = 40
moneyness1 = spot/strike
moneyness2=spot/strike
T =1
r = 0.06 
vol1 = 0.2
vol2 = 0.3
rho = 0.5
row = [moneyness1,moneyness2,T, r, vol1, vol2, rho]
yhat = predict(row, loaded_model3)*strike
print('Predicted American Min: %.3f' % yhat)
finish = datetime.datetime.now()
print (finish-start)
