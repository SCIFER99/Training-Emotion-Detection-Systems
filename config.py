# By: Tim Tarver

# This is the Config.py file for implementing the
# Real-Time Emotion Detection System

import os

# Initialize the path to the root folder where the dataset resides
# and the path to the train and test directory.

dataset_folder = 'Users\ttarv\OneDrive\Desktop\archive\test\angry'
train_directory = os.path.join(dataset_folder, "train")
test_directory = os.path.join(dataset_folder, "test")

# We initialize the amount of samples to use for training and validation

train_size = 0.90
value_size = 0.10

# Specify the batch size, total number of epochs and the learning rate

batch_size = 16
number_of_epochs = 50
learning_rate = 1e-1

