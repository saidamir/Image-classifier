import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import sys
from collections import OrderedDict


parse = argparse.ArgumentParser()
# if it's not specified then it is True
parse.add_argument('--gpu', action = 'store', default = 'gpu')
parse.add_argument('--save_dir', action = 'store', default = './checkpoint.pth')
parse.add_argument('--learning_rate', action = 'store', default = 0.001, type = float)
parse.add_argument('--epochs', action = 'store', default = 15, type = int)
parse.add_argument('--hidden_units', action = 'store', default = 600, type = int)
parse.add_argument('--dropout', action = 'store', default = 0.05, type = float)
parse.add_argument('--arch', action = 'store', default = 'vgg16')

parse.add_argument('data_dir', default = './flowers')

pa = parse.parse_args()
gpu = pa.gpu
save_dir = pa.save_dir
learning_rate = pa.learning_rate
epochs = pa.epochs
hidden_units = pa.hidden_units
dropout = pa.dropout
arch = pa.arch
data_dir = pa.data_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
test_transforms =transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader = torch.utils.data.DataLoader(validation_data, batch_size=32)


if arch == 'vgg13':
    model = models.vgg13(pretrained = True)
elif arch == 'vgg16':
    model = models.vgg16(pretrained = True)
else:
    print ('Architecture not supported')
    sys.exit(1)

for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate )
model.to(device);

print_every = 5
steps = 0
loss_show=[]

# change to cuda
model.to(device)

for e in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        steps += 1

        inputs,labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            vlost = 0
            accuracy=0


            for inputs2,labels2 in validloader:
                optimizer.zero_grad()

                inputs2, labels2 = inputs2.to(device) , labels2.to(device)
                model.to(device)
                with torch.no_grad():
                    outputs = model.forward(inputs2)
                    valid_lost = criterion(outputs,labels2)
                    ps = torch.exp(outputs).data
                    equality = (labels2.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

            valid_lost = valid_lost / len(validloader)
            accuracy = accuracy /len(validloader)



            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Train Loss: {:.3f}".format(running_loss/print_every),
                  "Validation Lost {:.3f}".format(valid_lost),
                   "Validation Accuracy: {:.3f}".format(accuracy))


            running_loss = 0

correct = 0
total = 0
model.to(device)


with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # Get probabilities
        outputs = model(images)
        # Turn probabilities into predictions
        _, predicted_outcome = torch.max(outputs.data, 1)
        # Total number of images
        total += labels.size(0)
        # Count number of cases in which predictions are correct
        correct += (predicted_outcome == labels).sum().item()

print(f"Test accuracy of model: {round(100 * correct / total,3)}%")

model.class_to_idx = train_data.class_to_idx
model.cpu
torch.save({'model' :arch,
            'classifier': model.classifier,
            'num_epochs': epochs,
            'hidden_layer1':hidden_units,
            'state_dict':model.state_dict(), #A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
            'class_to_idx':model.class_to_idx},
            save_dir)
