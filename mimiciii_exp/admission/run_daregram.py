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
from daregram import *
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


""" 
Read in the original dataframe
"""
output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/mimiciii")
print(f"Will save outputs to {output_dir}")

admid_diagnosis_df = pd.read_csv(os.path.join(output_dir, "ADMID_DIAGNOSIS.csv"), index_col=0, header=0, converters={'ICD codes': literal_eval})
print(admid_diagnosis_df)



suffix = None

source_count = 120
target_count = 100
iterations = 100
trans_metric = 'daregram'

# group_name = 'insurance'
# groups = ['Self_Pay', 'Private', 'Government', 'Medicare', 'Medicaid']

group_name = 'marital_status'
groups = ['MARRIED', 'SINGLE', 'WIDOWED', 'DIVORCED', 'SEPARATED']

for source in groups:
    for target in groups:
        if source == target:
            continue

        print(f"source is: {source}, target is: {target}")
        score_path = os.path.join(output_dir, f"{group_name}_{target}2{source}_{trans_metric}.csv")
        if os.path.exists(score_path):
            continue

        maes = []
        rmses = []
        for i in range(iterations):
            print("iteration:", i)
            selected_df = select_samples(admid_diagnosis_df, group_name, source, target, source_count, target_count)
            code_feature_name = 'ICD codes'
            label_name = 'duration'
            source_data, source_labels, target_data, target_labels = gen_code_feature_label(selected_df, group_name, source, target, code_feature_name, label_name)
            print("print data dimensions:", source_data.shape, source_labels.shape, target_data.shape, target_labels.shape)

            test_rmse, test_mae = run_daregram(source_data, source_labels, target_data, target_labels)
            maes.append(test_mae)
            rmses.append(test_rmse)

        print("rmses is:", rmses)
        print("maes is:", maes)
        save_results(rmses, maes, score_path)