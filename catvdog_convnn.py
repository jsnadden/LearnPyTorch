import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# globals
REBUILD_DATA = False
ALLOW_GPU = True
IMG_SIZE = 48

# hyperparameters
BATCH_SIZE = 100
EPOCHS = 1

print("----------------------------------------")

# query gpu availability
if torch.cuda.is_available() & ALLOW_GPU:
    DEVICE = torch.device("cuda:0")
    print("Running on GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on CPU")

#########################################
########## SOURCE & CLEAN DATA ##########
#########################################

class DogsVCats():
    cat_dir = "./PetImages/Cat"
    dog_dir = "./PetImages/Dog"

    labels = {cat_dir:0, dog_dir:1}

    data = []
    num_cats = 0
    num_dogs = 0

    def build_data(self):
        print("----------------------------------------")
        print("Loading images...")
        
        for label in self.labels:
            # tqdm gives us a progress bar as we load images
            for f in tqdm(os.listdir(label)):
                try:
                    # load and resize image
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    
                    # add in/out pair [pixel array, one-hot vector] to data_set
                    self.data.append([np.array(img), np.eye(2)[self.labels[label]]])

                    # count cats/dogs separately
                    if label == self.cat_dir:
                        self.num_cats += 1
                    elif label == self.dog_dir:
                        self.num_dogs += 1

                # skip past corrupted images
                except Exception as e:
                    pass

        print("Completed loading:", self.num_cats, "cats, and", self.num_dogs, "dogs were loaded.")

        np.random.shuffle(self.data)

        print("----------------------------------------")
        print("Serialising data...")

        with open("data.pkl", "wb") as f:
            pickle.dump(self.data, f)

        print("Complete!")

if REBUILD_DATA:
    dogsvcats = DogsVCats()
    dogsvcats.build_data()

print("----------------------------------------")
print("Loading data...")

# unpickle
data = np.load("data.pkl", allow_pickle=True)

print("Complete!")



#####################################
########## CONFIGURE MODEL ##########
#####################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 5, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 5, padding='same')

        # compute the size of the flattened output of the conv layers
        # (each pooling will halve the feature map dimensions)
        self.flattened_convs = int(128 * (IMG_SIZE / 2**3)**2)

        # dense layers
        self.fc1 = nn.Linear(self.flattened_convs, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # pass through convolutional layers
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        # flatten conv output
        x = x.view(-1, self.flattened_convs)
        # pass through dense layers
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

print("----------------------------------------")
print("Configuring model architecture...")

net = Net().to(DEVICE)

optimiser = optim.Adam(net.parameters(), lr=1e-3)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in data]).view(-1, IMG_SIZE, IMG_SIZE)
X = X / 255.0
y = torch.Tensor([i[1] for i in data])

# data split: 90% training / 10% testing
DATA_SIZE = len(X)
TEST_SIZE = int(DATA_SIZE * 0.1)

train_X = X[:-TEST_SIZE]
train_y = y[:-TEST_SIZE]

test_X = X[-TEST_SIZE:]
test_y = y[-TEST_SIZE:]

print("Complete!")

#################################
########## TRAIN MODEL ##########
#################################

print("----------------------------------------")
print("Training started...")

for epoch in range (EPOCHS):
    print("Epoch", epoch, ":")
    for i in tqdm(range(0, DATA_SIZE - TEST_SIZE, BATCH_SIZE)):

        # create a batch, send to GPU
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,IMG_SIZE,IMG_SIZE).to(DEVICE)
        batch_y = train_y[i:i+BATCH_SIZE].to(DEVICE)

        # backprop
        net.zero_grad()
        output = net(batch_X)
        loss = loss_function(output, batch_y)
        loss.backward()
        optimiser.step()
    print("Current loss value =", round(loss.item(), 5))

print("Training complete!")

####################################
########## VALIDATE MODEL ##########
####################################

print("----------------------------------------")
print("Validating...")

correct = 0

with torch.no_grad():
    for i in tqdm(range(TEST_SIZE)):
        target_class = torch.argmax(test_y[i]).to(DEVICE)
        output = net(test_X[i].view(-1,1,IMG_SIZE, IMG_SIZE).to(DEVICE))[0]
        predicted_class = torch.argmax(output)

        if predicted_class == target_class:
            correct += 1
        
print("Complete!")
print("Final accuracy:", round(correct/TEST_SIZE,3)*100, "%")

print("----------------------------------------")