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
import torch.nn as nn
import torch.optim as optim



output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/mimiciii")
print(f"Will save outputs to {output_dir}")


#################### Load data ####################

group_1_count = 120
group_2_count = 100
group_name = 'insurance'
group_1 = 'Self_Pay'
group_2 = 'Private'
input_name = 'ICD codes'
output_name = 'duration'

admid_diagnosis_df = pd.read_csv(os.path.join(output_dir, "ADMID_DIAGNOSIS.csv"), index_col=0, header=0, converters={'ICD codes': literal_eval})
selected_df = select_samples(admid_diagnosis_df, group_name, group_1, group_2, source_count=group_1_count, target_count=group_2_count)
source_data, source_labels, target_data, target_labels = gen_code_feature_label(selected_df, group_name, group_1, group_2, input_name, output_name)

n_components = 50
source_data, target_data = custom_train_reps_default(source_data, target_data, n_components)

 # Convert the numpy arrays into torch tensors
source_data_tensor = torch.tensor(source_data.astype(np.float32))
source_labels_tensor = torch.tensor(source_labels.astype(np.float32)).view(-1, 1)  # Reshaping for a single output feature
target_data_tensor = torch.tensor(target_data.astype(np.float32))
target_labels_tensor = torch.tensor(target_labels.astype(np.float32)).view(-1, 1)

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(source_data.shape[1], 1)  # Assuming source_data is 2D

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegressionModel()

# Define the loss criterion and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 20000
for epoch in range(num_epochs):
    # Convert numpy arrays to torch variables for gradient computation
    inputs = Variable(source_data_tensor)
    labels = Variable(source_labels_tensor)

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward
    optimizer.zero_grad()

    # Get output from the model, given the inputs
    outputs = model(inputs)

    # Get loss for the predicted output
    loss = criterion(outputs, labels)
    
    # Get gradients w.r.t to parameters
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model with training data
model.eval()
with torch.no_grad():  # We don't need gradients in the testing phase
    predicted_train = model(Variable(source_data_tensor)).data.numpy()
    train_loss = np.mean((predicted_train - source_labels) ** 2)
    print(f'Training Mean Squared Error: {train_loss}')

    # Similarly for testing data
    predicted_test = model(Variable(target_data_tensor)).data.numpy()
    test_loss = np.mean((predicted_test - target_labels) ** 2)
    print(f'Testing Mean Squared Error: {test_loss}')
