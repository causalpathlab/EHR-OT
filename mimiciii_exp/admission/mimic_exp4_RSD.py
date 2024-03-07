import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/OTTEHR/")
sys.path.append(f"/home/{user_id}/unbalanced_gromov_wasserstein/")
sys.path.append(f"/home/{user_id}/OTTEHR/competitors/")

from ast import literal_eval
from mimic_common import *
import numpy as np
import os
import ot
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from common import *
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


#################### Test using synthetic data ####################

import numpy as np

# Generating synthetic data
np.random.seed(0)  # For reproducibility

# Parameters
n_source_samples = 1000
n_target_samples = 2000
n_features = 100

# Synthetic feature data
source_data = np.random.rand(n_source_samples, n_features)
target_data = np.random.rand(n_target_samples, n_features)

# Synthetic labels (target variables), assuming a simple linear relation with some noise
weights = np.random.rand(n_features, 1)  # Random weights for our linear model
source_labels = source_data @ weights + np.random.rand(n_source_samples, 1) * 5  # Adding some noise
target_labels = target_data @ weights + np.random.rand(n_target_samples, 1) * 5  # Adding some noise

source_data.shape, source_labels.shape, target_data.shape, target_labels.shape

# Convert the numpy arrays into torch tensors
source_data_tensor = torch.tensor(source_data.astype(np.float32))
source_labels_tensor = torch.tensor(source_labels.astype(np.float32)).view(-1, 1)  # Reshaping for a single output feature
target_data_tensor = torch.tensor(target_data.astype(np.float32))
target_labels_tensor = torch.tensor(target_labels.astype(np.float32)).view(-1, 1)

n_components = 10


# Define the linear regression model with a feature extraction layer
class LinearRegressionModel(nn.Module):
    def __init__(self, input_features=source_data.shape[1], output_features=source_labels.shape[1], hidden_units=n_components):
        super(LinearRegressionModel, self).__init__()
        # Feature extraction layers
        self.extraction = nn.Linear(input_features, hidden_units)  # First hidden layer

        # Linear regression output layer
        self.output = nn.Linear(hidden_units, output_features)  # Output layer

    def forward(self, x):
        # Apply non-linear transformations for feature extraction
        print("x is:", x)
        print("weights of extraction layer is:", self.extraction.weight.data)
        print("x extraction is:", self.extraction(x))
        feature = self.extraction(x)

        # Linear output
        x = self.output(feature)
        print("weights of the output layer is:", self.output.weight.data)
        return feature, x

def RSD_BMP(source_feature, target_feature, RSD_coef = 0.001, BMP_coef = 0.1, eps=1e-6):
    def smooth_approximation(x):
        # Parameters for smooth transition
        k = 10  # Controls the sharpness of the transition
        epsilon = 1e-2  # Small value to avoid division by zero
        
        # Transition from 0 to x: sigmoid(k * x)
        y_start = F.sigmoid(k * x) * x
        
        # Transition from x to 1: (1 - sigmoid(k * (x - 1)))
        y_end = 1 - F.sigmoid(k * (x - 1))
        
        # Combine both parts
        y = y_start * y_end
        
        # Ensure output is smoothed at boundaries
        y = torch.where(x < epsilon, torch.zeros_like(x), y)  # Approximates y = 0 when x < 0
        y = torch.where(x > 1 - epsilon, torch.ones_like(x), y)  # Approximates y = 1 when x > 1
    
        return y

    def softplus_abs(x, beta=10):
        # A smooth approximation of the absolute value function
        return (torch.nn.functional.softplus(beta * x) + torch.nn.functional.softplus(-beta * x)) / (2 * beta)


    def add_noise(feature):
        eps = 1e-7
        # Normalize the matrix: subtract the mean and divide by the standard deviation of each feature
        mean = feature.mean(dim=0, keepdim=True)
        std = feature.std(dim=0, keepdim=True)
        matrix_norm = (feature - mean) / (std + eps)  # Adding a small value to prevent division by zero

        # Add small Gaussian noise to the matrix to improve stability
        noise_level = 1e-6  # Adjust based on your data and needs
        noise = torch.randn_like(matrix_norm) * noise_level
        noisy_feature = matrix_norm + noise
        return noisy_feature

    noisy_source_feature_t = add_noise(source_feature.t())
    noisy_target_feature_t = add_noise(target_feature.t())
    print("noisy_source_feature_t is:", noisy_source_feature_t)
    print("noisy_target_feature_t is:", noisy_target_feature_t)

    u_s, _, _ = torch.svd(noisy_source_feature_t)
    u_t, _, _ = torch.svd(noisy_target_feature_t)
    print("u_s is:", u_s)
    print("u_t is:", u_t)
    noisy_product = add_noise(torch.mm(u_s.t(), u_t))
    print("noisy_product is:", noisy_product)
    p_s, cosine, p_t = torch.svd(noisy_product)
    print("p_s is:", p_s)
    print("p_t is:", p_t)
    print("cospa is:", cosine)
    adjusted_cosine = torch.clamp(1 - torch.pow(cosine, 2), min=eps)
    sine = torch.sqrt(adjusted_cosine)
    soft_diff = softplus_abs(p_s - p_t)

    return RSD_coef*(torch.norm(sine, 1) + BMP_coef * torch.norm(soft_diff, 2))

# Instantiate the model
model = LinearRegressionModel()

# Define the loss criterion and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Convert numpy arrays to torch variables for gradient computation
    source_input_var = Variable(source_data_tensor)
    source_label_var = Variable(source_labels_tensor)
    target_input_var = Variable(target_data_tensor)

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward
    optimizer.zero_grad()

    # Get output from the model, given the inputs
    source_feature, source_outputs = model(source_input_var)
    target_feature, _ =  model(target_input_var)
    print("source_feature is:", source_feature)

    # Get loss for the predicted output
    regression_loss = criterion(source_outputs, source_label_var)
    rsd_bmp_loss = RSD_BMP(source_feature, target_feature)
    print("regression loss is:", regression_loss, "rsd_bmp_loss is:", rsd_bmp_loss)
    total_loss = regression_loss + rsd_bmp_loss
    
    # Get gradients w.r.t to parameters
    total_loss.backward()

   

    print("before optimization, print out model parameters")
    for name, param in model.named_parameters():
        print("name is:", name)
        print("param is:", param)
        print("param.grad is:", torch.nan_to_num(param.grad))

    # Clip graidents
    clip_value = 1e-6
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    print("after gradient clipping, print out model parameters")
    for name, param in model.named_parameters():
        print("name is:", name)
        print("param is:", param)
        print("param.grad is:", torch.nan_to_num(param.grad))

    # Update parameters
    optimizer.step()

    print("after optimiazation, print out model parameters")
    for name, param in model.named_parameters():
        print("name is:", name)
        print("param is:", param)
        print("param.grad is:", param.grad)

    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}')

    


# Evaluate the model with training data
model.eval()
with torch.no_grad():  # We don't need gradients in the testing phase
    _, predicted_train = model(Variable(source_data_tensor))
    predicted_train = predicted_train.data.numpy()
    # predicted_train = model(Variable(source_data_tensor)).data.numpy()
    train_loss = np.mean((predicted_train - source_labels) ** 2)
    print(f'Training Mean Squared Error: {train_loss}')

    # Similarly for testing data
    _, predicted_test = model(Variable(target_data_tensor))
    predicted_test = predicted_test.data.numpy()
    # predicted_test = model(Variable(target_data_tensor)).data.numpy()
    test_loss = np.mean((predicted_test - target_labels) ** 2)
    print(f'Testing Mean Squared Error: {test_loss}')
