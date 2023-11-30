import sys
sys.path.append("/home/wanxinli/OTTEHR/")

from ast import literal_eval
from common import *
from mimic_common import *
from multiprocess import Pool
import os
import ot
import ot.plot
import random
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import time


output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/mimic")
print(f"Will save outputs to {output_dir}")

""" 
Read in the original dataframe
"""
admid_diagnosis_df = pd.read_csv("../../outputs/mimic/ADMID_DIAGNOSIS.csv", index_col=0, header=0, converters={'ICD codes': literal_eval})
print(admid_diagnosis_df)

""" 
Print number of patients for each category
"""
print("female:", admid_diagnosis_df.loc[(admid_diagnosis_df['gender'] == 'F')].shape[0])
print("male:", admid_diagnosis_df.loc[(admid_diagnosis_df['gender'] == 'M')].shape[0])



""" 
Run multiple iterations using linear regression
"""
n_components = 50
score_path = os.path.join(output_dir, "exp4_MMD_linear_score.csv")
male_count = 120
female_count = 100

source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
    trans_target_maes, trans_target_mses, trans_target_rmses \
        = multi_proc_cts(n_components, admid_diagnosis_df, custom_train_reps_default, \
            male_count, female_count, trans_metric='MMD',model_func = linear_model.LinearRegression, iteration=100)

save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, \
    trans_target_maes, trans_target_mses, trans_target_rmses, score_path)
