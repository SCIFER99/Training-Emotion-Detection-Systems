# By: Tim Tarver

# Train.py script developed to begin training our emotion
# detection system

from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import classification_report
from torchvision.transforms import RandomCrop
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import config as cfg
from utils import EarlyStop
from utils import LRScheduler
from torchvision import transforms
from emotionNet import EmotionNet
from torchvision import datasets
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from torch.optim import SGD
import torch.nn as nn
import pandas as pd
import argparse
import torch
import math

# Initialize the argument parser and establish the arguments required

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Path to save the trained model')
parser.add_argument('-p', '--plot', type=str, help='Path to save the loss/accuracy plot')
args = vars(parser.parse_args())
 
# Configure the device to use for training the model, either gpu or cpu

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Current training device: {device}")

# Initialize a list of preprocessing steps to apply on each image during
# training/validation and testing

training_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    RandomHorizontalFlip(),
    RandomCrop((48, 48)),
    ToTensor()
])

test_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    ToTensor()
])

# Load all the images within the specified folder and apply different augmentation

training_data = datasets.ImageFolder(cfg.train_directory, transform=training_transform)
test_data = datasets.ImageFolder(cfg.test_directory, transform=test_transform)

# Extract the class labels and the total number of classes

classes = training_data.classes
number_of_classes = len(classes)
print(f"[INFO] Class labels: {classes}")

# Use train samples to generate train/validation set

number_training_samples = len(training_data)
training_size = math.floor(number_training_samples * cfg.TRAIN_SIZE)
value_size = math.ceil(number_training_samples * cfg.VAL_SIZE)
print(f"[INFO] Training Samples: {training_size} ...\t Validation samples: {value_size}...")

# Randomly split the training dataset into train and validation set

training_dataset, validation_dataset = random_split(training_data, [training_size, value_size])

# Modify the data transform applied towards the validation set

validation_dataset.dataset.transforms = test_transform

# Get the lables within the training set

training_classes = [label for _, label in training_dataset]

# Count each labels within each classes

class_count = Counter(training_classes)
print(f"[INFO] Total Sample: {class_count}")

# Compute and determine the weights to be applied on each category
# depending on the number of samples available

class_weight = torch.Tensor([len(training_classes) / c
                             for c in pd.Series(class_count).sort_index().values])

# Initialize a placeholder for each target image, and iterate via the training
# dataset, get the weights for each class and modify the default sample
# weight to its corresponding class weight already computed.

sample_weight = [0] * len(training_dataset)
for index, image, label in enumerate(training_dataset):
    weight = class_weight[label]
    sample_weight[index] = weight

# Define a sampler which randomly sample labels from the train dataset.

sampler = WeightedRandomSampler(weights=sample_weight, number_samples=len(training_dataset),
                                replacement=True)

# Load our own dataset and store each sample with their corresponding labels

train_data_loader = DataLoader(training_dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler)
value_data_loader = DataLoader(value_data, batch_size=cfg.BATCH_SIZE)
test_data_loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE)

# Initialize the training model and send it to our desired device.

model = EmotionNet(number_of_channels=1, number_of_classes=number_of_classes)
model_1 = model.to(device)

# Initialize our optimizer and loss function

optimizer = SGD(params=model_1.parameters(), lr=cfg.LR)
criterion = nn.CrossEntropyLoss()

# Initialize the learning rate scheduler and early stopping mechanism.

lr_scheduler = LRScheduler(optimizer)
early_stop = EarlyStop()

# Calculate the steps per epoch for training and validation set.

training_steps = len(train_data_loader.dataset) // cfg.BATCH_SIZE
value_steps = len(value_data_loader.dataset) // cfg.BATCH_SIZE

# Initialize a dictionary to save the training history.

training_history = {
    "training_accumulation": [],
    "training_loss": [],
    "value_accumulation": [],
    "value_loss": []
    }

# Iterate through the epochs

print(f"[INFO] Training the model...")
start_time = datetime.now()

for epoch in range(0, cfg.NUM_OF_EPOCHS):

    print(f"[INFO] epoch: {epoch + 1} / {cfg.NUM_OF_EPOCHS}")

    """
    Training the Model
    """
    # Set the model to training mode

    model_1.train()

    # Initialize the total training and validation loss and
    # the total number of correct predictions in both steps

    total_training_loss = 0
    total_value_loss = 0
    correct_training = 0
    correct_value = 0

    # Iterate through the training set.

    for data, target in train_data_loader:

        # Move the data into the device used for training,

        data, target = data.to(device), target.to(device)

        # Perform a forward pass and calculate the training loss

        predictions = model(data)
        loss = criterion(predictions, target)

        # If (not literally) zero gradients are accumulated from the
        # previous operation, perform a backward pass, and then update
        # the model parameters.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add the training loss and keep track of the number of correct
        # predictions.

        total_training_loss += loss
        correct_training += (predictions.argmax(1) == target).type(torch.float)

    """
    Validating the Model
    """
    model_1.eval() # Disable dropout and dropout layers

    # Prevent Pytorch form calculating the gradients, reducing
    # memory usage and speeding up the computation time (no back drop)

    with torch.set_grad_enabled(False):

        # Iterate through the validation set

        for data, target in value_data_loader:

            # Move the data into the device used for testing

            data, target = data.to(device), target.to(device)

            # Perform a forward pass and calculate the training loss

            predictions = model(data)
            loss = criteria(predictions, target)

            # Add the training loss and keep track of the number of
            # correct predictions.

            total_value_loss += loss
            correct_value += (predictions.argmax(1) == target).type(torch.float).sum().item()

    # Calculate the average training and validation loss

    average_training_loss = total_training_loss / training_steps
    average_value_loss = total_value_loss / value_steps

    # Calculate the training and validation accuracy
    
    correct_training = correct_training / len(train_data_loader.dataset)
    correct_value = correct_value / len(value_data_loader.dataset)

    # Print model training and validation records

    print(f"Training Loss: {average_training_loss:.3f} .. Training Accuracy: {correct_training:.3f}")
    print(f"Value Loss: {average_value_loss:.3f} .. Value Accuracy: {correct_value:.3f}", end='\n\n')

    # Update the training and validation records

    training_history['training_loss'].append(average_training_loss.cpu().detach().numpy())
    training_history['training_accumulation'].append(correct_training)
    training_history['value_loss'].append(average_value_loss.cpu().detach().numpy())
    training_history['value_accumulation'].append(correct_value)

    # Execute the learning rate scheduler and early stop

    validation_loss = average_value_loss.cpu().detach().numpy()
    lr_scheduler(validation_loss)
    early_stop(validation_loss)

    # Stop the training procedure due to no improvement while
    # validating the model

    if early_stop.early_stop_enabled:
        break

print(f"[INFO] Total Training Time: {datetime.now() - start_time}...")

# Move the model back to CPU and save the trained model to disk

if device == "cuda":

    model_1 = model_1.to("cpu")

# Then save it.

torch.save(model_1.state_dict(), args['model'])

# Plot the training loss and accuracy overtime

plt.style.use("ggplot")
plt.figure()
plt.plot(training_history['training_accumulation'], label='training_accumulation')
plt.plot(training_history['value_accumulation'], label='value_accumulation')
plt.plot(training_history['training_loss'], label='training_loss')
plt.plot(training_history['value_loss'], label='value_loss')
plt.ylabel('Loss/Accuracy')
plt.xlabel("#Number of Epochs")
plt.title('Training Loss and Accuracy on FER2013')
plt.legend(loc='upper right')
plt.savefig(args['plot'])

# Evaluate the model based on the test set

model_1 = model_1.to(device)

with torch.set_grad_enabled(False):

    # set the evaluation model

    model_1.eval()

    # Initialize a list to keep track of our predictions

    predictions = []

    # Iterate through the test set

    for data, _ in test_data_loader:

        # Move the data into the device used for testing

        data_1 = data.to(device)

        # Perform a forward pass and calculate the training class

        output = model(data_1)
        output_1 = output.argmax(axis=1).cpu().numpy()
        predictions.extend(output_1)

# Evaluate the Network

print("[INFO] Evaluating Network...")
actual = [label for _, label in test_data]
print(classification_report(actual, predictions, target_names = test_data.classes))


