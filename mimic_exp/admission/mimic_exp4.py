import sys
sys.path.append("/home/wanxinli/EHR-OT/")

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


output_dir = os.path.join(os.path.expanduser("~"), f"EHR-OT/outputs/mimic")
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
Train deep patient model and generate representations for targets and sources
"""

def custom_train_reps(source_features, target_features, n_components, pca_explain=False):
    """ 
    Customized training algorithm for generating target representations and source representations

    :param bool pca_explain: print the explained variance of each components
    
    :returns: target representations, source representations
    """
    source_pca = PCA(n_components=n_components)
    source_reps = source_pca.fit_transform(source_features)

    target_pca = PCA(n_components=n_components)
    target_reps = target_pca.fit_transform(target_features)

    if pca_explain:
        source_exp_var = source_pca.explained_variance_ratio_
        source_cum_sum_var = np.cumsum(source_exp_var)
        target_exp_var = target_pca.explained_variance_ratio_
        target_cum_sum_var = np.cumsum(target_exp_var)
        print("Cummulative variance explained by the source PCA is:", source_cum_sum_var[-1])
        print("Cummulative variance explained by the target PCA is:", target_cum_sum_var[-1])

    return source_reps, target_reps


""" 
Run multiple iterations using linear regression
"""
n_components = 50
score_path = os.path.join(output_dir, "exp4_linear_score_"+str(n_components)+".csv")
male_count = 120
female_count = 100

# source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
#     trans_target_maes, trans_target_mses, trans_target_rmses \
#         = multi_proc_cts(n_components, admid_diagnosis_df, custom_train_reps, \
#             male_count, female_count, model_func = linear_model.LinearRegression, iteration=10)

# save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, \
#     trans_target_maes, trans_target_mses, trans_target_rmses, score_path)
# print(res)

# """ 
# Run multiple iterations using Poisson regression
# """
# score_path = os.path.join(output_dir, "exp4_poisson_score_"+str(n_components)+".csv")
# male_count = 120
# female_count = 100

# multi_proc_cts(score_path, n_components, admid_diagnosis_df, custom_train_reps, \
#     male_count, female_count, model_func = linear_model.PoissonRegressor, iteration=100)

score_path = os.path.join(output_dir, "exp4_SVR_score_"+str(n_components)+".csv")

source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
    trans_target_maes, trans_target_mses, trans_target_rmses \
        = multi_proc_cts(n_components, admid_diagnosis_df, custom_train_reps, \
            male_count, female_count, model_func = SVR, iteration=10)

save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, \
    trans_target_maes, trans_target_mses, trans_target_rmses, score_path)