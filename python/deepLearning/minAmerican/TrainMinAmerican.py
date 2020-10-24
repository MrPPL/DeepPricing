


#model = NeuralNet(input_size, hidden_size1, l1, l2, outputSize)
#writer = SummaryWriter('runs/AmericanPutMin/100K')
#writer.add_graph(model, dataset.x_data)
#train the model
#train_model(hParameters, train_dl, valid_dl)

#analysis = tune.run(
#    train_mnist, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

#print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

# Get a dataframe for analyzing trial results.
#df = analysis.dataframe()
#writer.flush() # all pending method events has been written to disk
#writer.close()
# evaluate the model
# calculate mse
#actuals, predictions = evaluate_model(test_dl, model)
#mse = mean_squared_error(actuals, predictions)
#print('MSE: %.6f, RMSE: %.6f' % (mse, np.sqrt(mse)))

# save model
#torch.save(model.state_dict(), "/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepHedging/python/deepLearning/Models/hirsaModel.pth")
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm #follow progress in training

##########
# Dataset Class
##########
class Dataset(nn.Module):
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
    def get_splits(self, n_valid=0.2):
        # determine sizes
        valid_size = round(n_valid * len(self.x_data))
        train_size = len(self.x_data) - valid_size
        # calculate the split
        return random_split(self, [train_size, valid_size])

######################
# prepare the dataset
####################
def prepare_data(dataset, batchsize):
    # calculate split
    train, valid = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=batchsize, shuffle=True)
    valid_dl = DataLoader(valid, batch_size=batchsize, shuffle=True)
    return train_dl, valid_dl 
    
#########################
# Design Model
###################
#fully connected Neural net
class Net(nn.Module):
    def __init__(self, input_size, l1, l2, l3, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, l1)
        self.leaky_relu_1 = nn.LeakyReLU()
        self.l2 = nn.Linear(l1, l2)
        self.leaky_relu_2 = nn.LeakyReLU()
        self.l3 = nn.Linear(l2, l3)
        self.leaky_relu_3 = nn.LeakyReLU()
        self.l4 = nn.Linear(l3, output_size)
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
    

def train_MLPs(config, checkpoint_dir=None, data_dir=None, dataset=Dataset("./deepLearning/minAmerican/data/100KAmerMinPut.csv")):
    model = Net(input_size=1, l1=config["l1"], l2=config["l2"], l3=config["l3"], output_size=1)
    device = "cpu"
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, validset = prepare_data(dataset, config["batch_size"])
    #enumereate epoch
    for epoch in range(10):
        # Training 
        epoch_loss = 0
        running_loss = 0
        epoch_steps=0
        loop = tqdm(enumerate(trainset), total=len(trainset), leave=False) #bar progress when trainin
        for i, (X, y) in loop:  #one batch of samples       
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad() # zero the gradient buffer
            #forward pass and loss
            y_predicted = model(X)
            loss = criterion(y_predicted,y) #loss
            #writer.add_scalar("Loss/train", loss, epoch)
            # Backward and optimize
            loss.backward()
            optimizer.step() #does weight update
            # show describtion of each loop
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss = loss.item())
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
            # accumulate loss
            epoch_loss += loss.item()

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        
        val_epoch_loss = 0
        loop1 = tqdm(enumerate(validset), total=len(validset), leave=False) #bar progress when validation
        for i, (X, y) in loop1:  #one batch of samples 
            with torch.no_grad():
                X,y = X.to(device), y.to(device)
                optimizer.zero_grad() # zero the gradient buffer
                #forward pass and loss
                y_predicted = model(X)
                loss = criterion(y_predicted,y) #loss
                loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
                loop.set_postfix(loss = loss.item())

                # accumulate loss
                val_epoch_loss += loss.item()
                
                val_loss += loss.cpu().numpy()
                val_steps += 1

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.9f}')
        #writer.add_scalar("Loss/Train", epoch_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.9f}')
        #writer.add_scalar("Loss/Validation", val_epoch_loss, epoch)
        #writer.add_hparams({'lr': learning_rate},{'Loss Train': epoch_loss, 'Loss Validation': val_epoch_loss })
        
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    dataset = Dataset("./deepLearning/minAmerican/data/100KAmerMinPut.csv")
    data_dir = os.path.abspath("./deepLearning/minAmerican/data")
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l3": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train_MLPs),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trained_model = Net(input_size=1,l1=best_trial.config["l1"], l2=best_trial.config["l2"], l3=best_trial.config["l3"], output_size=1)
    device = "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

main()