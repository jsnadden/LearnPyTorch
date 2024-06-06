import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms, datasets

# poll if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



###########################################
########## SOURCE/CLEAN OUR DATA ##########
###########################################

# download mnist dataset, splitting into testing and training subsets
train = datasets.MNIST("", train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
# load data into memory
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)



###############################################
########## CONFIGURE NN ARCHITECTURE ##########
###############################################

class Net(nn.Module):
    def __init__(self):
        # inherit init function from parent
        super().__init__()

        # configure layers
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
    
    # feedforward function (choose activations/loss function)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

# instantiate our NN
net = Net()

# configure optimiser (fixing a learning rate)
optimiser = optim.Adam(net.parameters(), lr=1e-3)



##########################################
########## TRAIN THE NEURAL NET ##########
##########################################

num_epochs = 3

for epoch in range(num_epochs):
    for batch in trainset:
        X,y = batch

        # if you don't zero the gradient, it will accumulate changes
        net.zero_grad()

        # feedforward data through network
        output = net(X.view(-1,28*28))

        # compute loss
        # since our output data are categorical, but represented as an integer, we use nll_loss
        # if it was one-hot encoded, we would want to use mse_loss instead
        loss = F.nll_loss(output, y)

        # backpropagate derivatives
        loss.backward()

        # perform gradient descent step
        optimiser.step()

    # print loss at the end of each epoch
    print(loss)