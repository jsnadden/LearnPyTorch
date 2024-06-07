import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# this is so ugly, and in desperate need of cleaning up...
# Python really doesn't inspire best practices like my beloved C++

# globals
REBUILD_DATA = False
ALLOW_GPU = True
IMG_SIZE = 48
BATCH_SIZE = 100
EPOCHS = 30

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
TRAIN_SIZE = DATA_SIZE - TEST_SIZE

train_X = X[:-TEST_SIZE]
train_y = y[:-TEST_SIZE]

test_X = X[-TEST_SIZE:]
test_y = y[-TEST_SIZE:]

MODEL_NAME = f"model-{int(time.time())}"

print("Complete!")



def feedthrough(X, y, train=False):
    if train:
        net.zero_grad()
    
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    accuracy = matches.count(True) / len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimiser.step()
    
    return accuracy, loss.item()

def test(size=32):
    random_index = np.random.randint(TEST_SIZE - size)

    X = test_X[random_index:random_index+size]
    y = test_y[random_index:random_index+size]

    with torch.no_grad():
        val_acc, val_loss = feedthrough(X.view(-1,1,IMG_SIZE,IMG_SIZE).to(DEVICE), y.to(DEVICE), False)
    
    return val_acc, val_loss

def train():
    print("----------------------------------------")
    print("Training started...")

    start_time = round(float(time.time()), 3)

    with open(f"{MODEL_NAME}.log", "a") as f:
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch}:")
            for i in tqdm(range(0, TRAIN_SIZE, BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,IMG_SIZE,IMG_SIZE).to(DEVICE)
                batch_y = train_y[i:i+BATCH_SIZE].to(DEVICE)

                acc, loss = feedthrough(batch_X, batch_y, train=True)
                
                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    time_since_start = round(float(time.time()), 3) - start_time
                    f.write(f"{time_since_start},{round(acc, 2)},{round(loss, 4)},{round(val_acc, 2)},{round(val_loss, 4)}\n")

    print("Training complete!")

train()



headers = ["time", "train_acc", "train_loss", "val_acc", "val_loss"]
df = pd.read_csv(f"{MODEL_NAME}.log",names=headers)

smoothing = 15
df['rolling_ta'] = df['train_acc'].rolling(smoothing).mean()
df['rolling_tl'] = df['train_loss'].rolling(smoothing).mean()
df['rolling_va'] = df['val_acc'].rolling(smoothing).mean()
df['rolling_vl'] = df['val_loss'].rolling(smoothing).mean()

t = df["time"]
ta = df["rolling_ta"]
tl = df["rolling_tl"]
va = df["rolling_va"]
vl = df["rolling_vl"]

plt.plot(t, ta)
plt.plot(t, tl)
plt.plot(t, va)
plt.plot(t, vl)
plt.show()

print("----------------------------------------")