import sys
sys.path.append("/home/wanxinli/EHR-OT/")

from ast import literal_eval
from common import *
from mimic_common import *
from multiprocess import Pool
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from statistics import mean, median
from tca import *


output_dir = os.path.join(os.path.expanduser("~"), f"EHR-OT/outputs/mimic")
print(f"Will save outputs to {output_dir}")

""" 
Read in the original dataframe
"""
admid_diagnosis_df = pd.read_csv("../../outputs/mimic/ADMID_DIAGNOSIS.csv", index_col=0, header=0, converters={'ICD codes': literal_eval})
# print(admid_diagnosis_df)

source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
    trans_target_maes, trans_target_mses, trans_target_rmses = \
    multi_proc_cts_tca(admid_diagnosis_df, custom_train_reps_default, linear_model.LinearRegression, n_times = 100)

tca_score_path = os.path.join(output_dir, "exp4_tca_linear_score.csv")
save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, \
    trans_target_maes, trans_target_mses, trans_target_rmses, tca_score_path)

# source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
#     trans_target_maes, trans_target_mses, trans_target_rmses = \
#     multi_proc_cts_tca(admid_diagnosis_df, custom_train_reps_default, linear_model.PoissonRegressor, n_times = 100)

# tca_score_path = os.path.join(output_dir, "exp4_tca_poisson_score.csv")
# save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, \
#     trans_target_maes, trans_target_mses, trans_target_rmses, tca_score_path)