import sys
sys.path.append("/home/wanxinli/OTTEHR/")
from sklearn.svm import SVR
from ast import literal_eval
from common import *
from mimic_common import *
import os


output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/mimic")
print(f"Will save outputs to {output_dir}")

""" 
Read in the original dataframe
"""
admid_diagnosis_df = pd.read_csv("../../outputs/mimic/ADMID_DIAGNOSIS.csv", index_col=0, header=0, converters={'ICD codes': literal_eval})
# print(admid_diagnosis_df)

n_neighbors = 20
source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
    trans_target_maes, trans_target_mses, trans_target_rmses = \
    multi_proc_cts_NN(admid_diagnosis_df, linear_model.LinearRegression, n_neighbors, n_times = 10)

nn_score_path = os.path.join(output_dir, "exp4_NN_linear_score.csv")
save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, \
    trans_target_maes, trans_target_mses, trans_target_rmses, nn_score_path)

# n_neighbors = 1

# source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
#     trans_target_maes, trans_target_mses, trans_target_rmses = \
#     multi_proc_cts_NN(admid_diagnosis_df, SVR, n_neighbors, n_times = 10)

# nn_score_path = os.path.join(output_dir, "exp4_nn_SVR_score.csv")
# save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, \
#     trans_target_maes, trans_target_mses, trans_target_rmses, nn_score_path)