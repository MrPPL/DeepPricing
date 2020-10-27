
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as pyplot
from numpy import vstack
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm #follow progress in training
from torch.utils.data import random_split
#read file
import sys 
import os
sys.path.append(os.path.abspath("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepLearning/minAmerican"))
import earlyStop

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
    
#########################
# Design Model
###################
#fully connected Neural net
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
        self.l4 = nn.Linear(hidden_size3, output_size)
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
######################
# prepare the dataset
####################
def prepare_data(dataPath):
    # calculate split
    train, valid = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=64, shuffle=True)
    valid_dl = DataLoader(valid, batch_size=1024, shuffle=False)
    return train_dl, valid_dl

def train_model(train_dl, valid_dl, model):
    #loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    es = earlyStop.EarlyStopping(patience=5)
    #enumereate epoch
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n_samples = 0
        loop = tqdm(enumerate(train_dl), total=len(train_dl), leave=False) #bar progress when trainin
        loop1 = tqdm(enumerate(valid_dl), total=len(valid_dl), leave=False) #bar progress when validation
        for i, (X, y) in loop:  #one batch of samples       
            optimizer.zero_grad() # zero the gradient buffer
            #forward pass and loss
            y_predicted = model(X)
            loss = criterion(y_predicted,y) #loss
            # Backward and optimize
            loss.backward()
            optimizer.step() #does weight update
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss = loss.item())

            # accumulate loss
            total_loss += loss.item() * X.shape[0]
            n_samples += X.shape[0]
        total_loss /= n_samples
        with torch.no_grad():
            model.eval()
            n_samples = 0
            total_val_loss = 0
            for i, (X, y) in loop1:  #one batch of samples 
                optimizer.zero_grad() # zero the gradient buffer
                #forward pass and loss
                y_predicted = model(X)
                loss = criterion(y_predicted,y) #loss
                loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
                loop.set_postfix(loss = loss.item())

                # accumulate loss
                total_val_loss += loss.item() * X.shape[0]
                n_samples += X.shape[0]
            
        total_val_loss /= n_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss:.9f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {total_val_loss:.9f}')
        
        if es.step(total_val_loss):
            break  # early stop criterion is met, we can stop now
    
#prepare data
dataPath = "./deepLearning/minAmerican/data/300KAmerMinPut.csv"
dataset = EuroParDataset(dataPath)
train_dl, valid_dl = prepare_data(dataPath)
print(len(train_dl.dataset), len(valid_dl.dataset))
#define network
#hyperparameters
input_size = dataset.x_data.shape[1]
hidden_size1 = 120
hidden_size2 = 120
hidden_size3 = 120
outputSize = dataset.y_data.shape[1]
num_epochs = 100
batchSize = 64
learning_rate = 0.001

model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, outputSize)
#train the model
train_model(train_dl, valid_dl, model)

# save model
torch.save(model.state_dict(), "/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepLearning/Models/ModelAM_Min1.pth")



