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
        xy = np.loadtxt("./deepLearning/hirsa19/data/mediumCEuroData.csv", delimiter=',', dtype=np.float32, skiprows=1)
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
hidden_size1 = 120
hidden_size2 = 120
hidden_size3 = 120
outputSize = 1
num_epochs = 10
batchSize = 64
learning_rate = 0.01
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

model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, outputSize)

#loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
#enumereate epoch
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



##############
# Evaluate Model
###############
from numpy import vstack
from sklearn.metrics import mean_squared_error
predictions, actuals = list(), list()
for i, (inputs, targets) in enumerate(validation_loader):
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

#model performance
# calculate mse
mse = mean_squared_error(actuals, predictions)
print('MSE: %.6f, RMSE: %.6f' % (mse, np.sqrt(mse)))

# Plot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['agg.path.chunksize']=200000
plt.plot(predictions, actuals, 'b')
plt.show()


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

row = [0.9,0.02, 0.5, 1]
yhat = predict(row, model)
print('Predicted: %.3f' % yhat)

    