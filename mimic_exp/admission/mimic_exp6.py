#!/usr/bin/env python
# coding: utf-8

# Transfer learning between different admission types (EMERGENCY vs ELECTIVE)

# In[1]:


from ast import literal_eval
from collections import Counter
import pandas as pd


admid_diagnosis_df = pd.read_csv("/home/wanxinli/EHR-OT/outputs/mimic/ADMID_DIAGNOSIS.csv", header=0, index_col=0,  converters={"ICD codes": literal_eval})
adm_types = list(admid_diagnosis_df['adm_type'])
print(Counter(adm_types))


# In[2]:


import sys
sys.path.append("/home/wanxinli/EHR-OT/")

from mimic_common import *
from multiprocess import Pool
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import time


# In[3]:


output_dir = os.path.join(os.path.expanduser("~"), f"EHR-OT/outputs/mimic")
print(f"Will save outputs to {output_dir}")


# In[4]:


"""
Train deep patient model and generate representations for targets and sources
"""

def custom_train_reps(target_features, source_features, n_components, pca_explain=False):
    """ 
    Customized training algorithm for generating target representations and source representations

    :param bool pca_explain: print the explained variance of each components
    
    :returns: target representations, source representations
    """
    source_pca = PCA(n_components=n_components)
    target_pca = PCA(n_components=n_components)
    target_reps = target_pca.fit_transform(target_features)
    source_reps = source_pca.fit_transform(source_features)

    if pca_explain:
        source_exp_var = source_pca.explained_variance_ratio_
        source_cum_sum_var = np.cumsum(source_exp_var)
        target_exp_var = target_pca.explained_variance_ratio_
        target_cum_sum_var = np.cumsum(target_exp_var)
        print("Cummulative variance explained by the source PCA is:", source_cum_sum_var)
        print("Cummulative variance explained by the target PCA is:", target_cum_sum_var)

    return target_reps, source_reps


# In[5]:


def multi_proc_parallel(score_path, n_components, label_code, custom_train_reps, \
        male_count, female_count, iteration=20):
    """ 
    Code cannot be parallized when passing the dataframe (full_df) as a parameter
    Hence, cannot be put into mimic_common.py
    """
    
    p = Pool(10)

    # note: the following line cannnot be used for parallelization either
    # admid_diagnosis_df = pd.read_csv("../../outputs/mimic/ADMID_DIAGNOSIS.csv", index_col=0, header=0, converters={'ICD codes': literal_eval})

    def iteration_wrapper(iter):
        """ 
        Wrapper function for one iteration, returns result statistics, for parallel computing

        :param int iter: the current iteration
        """
        print(f"iteration: {iter}\n")
        cur_res = entire_proc_binary(n_components, "adm_type", "ELECTIVE",  "EMERGENCY", label_code, admid_diagnosis_df, custom_train_reps, 
                    male_count = male_count, female_count = female_count, transfer_score=True)
        
        return cur_res

    res = p.map(iteration_wrapper, np.arange(0, iteration, 1))
    res_df = pd.DataFrame(res, columns = ['target_accuracy', 'target_precision', 'target_recall', 'target_f1', \
                                          'source_accuracy', 'source_precision', 'source_recall', 'source_f1', \
                                            'trans_source_accuracy', 'trans_source_precision', 'trans_source_recall', 'trans_source_f1', 'transfer_score', 'original_score'])
    res_df.to_csv(score_path, index=False, header=True)
    return res



# In[6]:


""" 
Run the entire proc for all response (i.e., label_code) 
Responses are selected by select_codes.ipynb and saved in ../../outputs/mimic/selected_summary_mimic.csv
"""

n_components = 20
male_count = 120
female_count = 50
label_code_path = os.path.join(output_dir, "selected_summary_mimic.csv")
label_code_df = pd.read_csv(label_code_path, header=0, index_col=None)
label_codes = list(label_code_df['ICD code'])[:1]
print("label_codes are:", label_codes)
for label_code in label_codes:
    start_time = time.time()
    print(f"label code {label_code} started")
    score_path = os.path.join(output_dir, f"exp6_{label_code}_score.csv")
    multi_proc_parallel(score_path, n_components, label_code, custom_train_reps, \
            male_count, female_count, iteration=100)
    end_time = time.time()
    print(f"runtime for {label_code} is: {end_time-start_time}")


# In[ ]:


# label_code = "008.45"
# cur_res = entire_proc_binary(n_components, "adm_type",  "ELECTIVE", "EMERGENCY", label_code, admid_diagnosis_df, custom_train_reps, 
#                     male_count=male_count, female_count = female_count, transfer_score=True)


# In[ ]:


# cur_res


# In[ ]:




