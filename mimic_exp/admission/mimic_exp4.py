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

suffix = None
group_name = 'gender'
group_1 = 'M'
group_2 = 'F'

# suffix = None
# group_name = 'marital_status'
# group_1 =  'MARRIED'
# group_2 = 'SINGLE'

# suffix = None
# group_name = 'marital_status'
# group_1 = 'SINGLE'
# group_2 = 'MARRIED'


# Select a subset of the data (newborn)
# admid_diagnosis_df = admid_diagnosis_df[admid_diagnosis_df['adm_type'] == 'NEWBORN']
# suffix = "newborn"

# suffix = None
# group_name = 'insurance'
# group_1 =  'Private'
# group_2 =  'Self_Pay'

group_1_count = 120
group_2_count = 100
trans_metric = 'OT'

score_path = os.path.join(output_dir, f"exp4_{group_name}_{group_2}2{group_1}_{trans_metric}.csv")
if suffix is not None:
    score_path = os.path.join(output_dir, f"exp4_{group_name}_{group_2}2{group_1}_{trans_metric}_{suffix}.csv")


source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
    trans_target_maes, trans_target_mses, trans_target_rmses \
        = multi_proc_cts(n_components, admid_diagnosis_df, custom_train_reps, group_name, group_1, group_2, \
            group_1_count, group_2_count, trans_metric=trans_metric, model_func = linear_model.LinearRegression, iteration=100, equity=True, suffix=suffix)

save_scores_cts(source_maes, source_mses, source_rmses,  target_maes, target_mses, target_rmses, \
    trans_target_maes, trans_target_mses, trans_target_rmses, score_path)
