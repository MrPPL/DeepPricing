'''
Video explanation: https://youtu.be/RLqsxWaQdHE
'''

# Imports
import torch
import torchvision
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader, Dataset # Gives easier dataset managment and creates mini batches
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import numpy as np # standard library for numpy
from torch.utils.data import random_split # Split data into training and validation set.
#read file
import sys 
import os
sys.path.append(os.path.abspath("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/python/deepLearning/minAmerican"))
import earlyStop

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
        valid_size = round(n_valid * len(self.x_data))
        train_size = len(self.x_data) - valid_size
        # calculate the split
        return random_split(self, [train_size, valid_size])

######################
# prepare the dataset
####################
def prepare_data(batchSize):
    # calculate split
    train, valid = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=batchSize, shuffle=True)
    valid_dl = DataLoader(valid, batch_size=32, shuffle=False)
    return train_dl, valid_dl

#prepare data
dataPath = "./deepLearning/hirsa19/data/1KCEuroData.csv"
dataset = EuroParDataset(dataPath)
#define network
#hyperparameters
input_size = dataset.x_data.shape[1]
hidden_size1 = 120
hidden_size2 = 120
hidden_size3 = 120
outputSize = dataset.y_data.shape[1]
num_epochs = 10

# To do hyperparameter search, include more batch_sizes you want to try
# and more learning rates!
batch_sizes = [64, 256, 1024]
learning_rates = [0.0001]
dataPath = ["./deepLearning/hirsa19/data/1KCEuroData.csv","./deepLearning/hirsa19/data/10KCEuroData.csv","./deepLearning/hirsa19/data/100KCEuroData.csv","./deepLearning/hirsa19/data/300KCEuroData.csv","./deepLearning/hirsa19/data/1MCEuroData.csv"]
#dataPath = ["./deepLearning/minAmerican/data/100KAmerMinPut.csv"]
#dataset = EuroParDataset(dataPath[0])


for DataPath in dataPath:
    dataset = EuroParDataset(DataPath)
    for batchSize in batch_sizes:
        for learning_rate in learning_rates:
            print("DataSet", DataPath[28:], "learning rate", learning_rate)
            es = earlyStop.EarlyStopping(patience=5)
            # Initialize network
            train_loader, valid_loader = prepare_data(batchSize)
            model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, outputSize)
            model.train()
            criterion = nn.MSELoss() 
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # Define Scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, verbose=True
            )
            tb = SummaryWriter(f'runs/EuroC1/batch_size={batchSize} lr={learning_rate} dataset={DataPath[28:]}')

            # Visualize model in TensorBoard
            features, _ = next(iter(train_loader))
            tb.add_graph(model, features)
            tb.close()

            N_EPOCHS = 100

            for epoch in range(N_EPOCHS):
                # Train
                model.train()  # IMPORTANT
                
                total_loss, n_samples = 0.0, 0
                for batch_i, (X, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    y_ = model(X)
                    loss = criterion(y_, y)
                    loss.backward()
                    optimizer.step()

                    # Statistics
                    #print(
                    #     f"Epoch {epoch+1}/{N_EPOCHS} |"
                    #     f"  batch: {batch_i} |"
                    #     f"  batch loss:   {loss.item():0.3f}"
                    #)
                    total_loss += loss.item() * X.shape[0]
                    n_samples += X.shape[0]
                
                mean_loss = total_loss / n_samples
                print(
                    f"Epoch {epoch+1}/{N_EPOCHS} |"
                    f"  train loss: {mean_loss:9.3f} |"
                )
                tb.add_scalar('Training Loss', mean_loss, epoch)
                
                for name, param in model.named_parameters():
                    tb.add_histogram(name, param, epoch)
                    tb.add_histogram(f'{name}.grad', param.grad, epoch)
                    
                
                # Eval
                model.eval()  # IMPORTANT
                
                total_loss, n_samples = 0.0, 0
                with torch.no_grad():  # IMPORTANT
                    for X, y in valid_loader:
                                
                        y_ = model(X)
                        # Statistics
                        loss = criterion(y_, y)
                        total_loss += loss.item() * X.shape[0]
                        n_samples += X.shape[0]
                    
                        # Plot things to tensorboard
                        #tb.add_histogram('leaky_relu4', model.l4.weight)
                        #tb.add_scalar('Training loss', loss, global_step=step)

                if es.step(total_loss/n_samples):
                    break  # early stop criterion is met, we can stop now
                
                print(
                    f"Epoch {epoch+1}/{N_EPOCHS} |"
                    #f"  valid loss: {total_loss / n_samples:9.3f} |"
                )
                tb.add_scalar('Validation Loss', total_loss/n_samples, epoch)
                #tb.add_hparams({'lr': learning_rate, 'bsize': batchSize}, 
                #       {'loss': total_loss/n_samples})
            
            tb.add_hparams({'DataSet': DataPath[28:],'lr': learning_rate, 'bsize': batchSize},{'loss': mean_loss})
            tb.add_hparams({'DataSet': DataPath[28:],'lr': learning_rate, 'bsize': batchSize},{'val. loss': total_loss/n_samples})

