import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as pyplot

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches
# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us
# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class EuroParDataset(Dataset):
    def __init__(self):
        # Initialize data, download, etc.
        xy = np.loadtxt("./deepLearning/data/mediumEuroData.csv", delimiter=',', dtype=np.float32, skiprows=1)
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


# create dataset
dataset = EuroParDataset()
# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
print(features, labels)

#hyperparameters
n_samples, n_features = len(dataset), len(features)
input_size = n_features
hidden_size1 = 100
hidden_size2 = 100
hidden_size3 = 100
hidden_size4 = 100
outputSize = 1
num_epochs = 10
batchSize = 64
learning_rate = 0.001
validation_split = 0.2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
indices = list(range(n_samples))
split = int(np.floor(validation_split * n_samples))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
#use dataloader to effectice minibatching
train_loader = DataLoader(dataset=dataset,
                          batch_size=batchSize,
                          num_workers=2,
                          sampler=train_sampler)

validation_loader = DataLoader(dataset=dataset,
                               batch_size=batchSize, 
                               num_workers=2, 
                               sampler=valid_sampler)

#Design model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.leaky_relu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.l5 = nn.Linear(hidden_size4, output_size)
    
    def forward(self,x):
        out = self.l1(x)
        out = self.leaky_relu(out)
        out = F.elu(self.l2(out))
        out= self.l3(out)
        out = self.leaky_relu(out)
        out = F.elu(self.l4(out))
        out = self.l5(out)
        return out

model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, outputSize)

#loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (X, y) in enumerate(train_loader):  
        #forward pass and loss
        y_predicted = model(X)
        loss = criterion(y_predicted,y)
        
        # Backward and optimize
        optimizer.zero_grad() # zero the gradient buffer
        loss.backward()
        optimizer.step() #does weight update
        
        if (i+1) % 5 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse

    