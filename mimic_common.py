import sys
sys.path.append("/home/wanxinli/EHR-OT/")

from datetime import datetime
import copy
from common import *
from collections import Counter
from ast import literal_eval
from MMD import *
import numpy as np
from NN import *
import matplotlib.pyplot as plt
from multiprocess import Pool
import os
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
    f1_score, mean_absolute_error, mean_squared_error, \
    mutual_info_score, normalized_mutual_info_score
from scipy.stats import entropy
from TCA import *

mimic_output_dir = "/home/wanxinli/EHR-OT/outputs/mimic"
mimic_data_dir = "/home/wanxinli/EHR-OT/mimic_exp/mimiciii"

# def update_codes(df):
#     """ 
#     Update code in dataframe, the new code starts from 0.
#     We use -1 to denote NA code in later analysis

#     returns
#         - a new dataframe with the updated codes
#         - total number of unique codes in df
#     """
#     new_code_dict = {} # mapping from old code to new code
#     for index, row in df.iterrows():
#         cur_codes = row['ICD codes']
#         new_codes = []
#         for code in cur_codes:
#             if code not in new_code_dict:
#                 new_code_dict[code] = len(new_code_dict) # add a new entry
#             new_codes.append(new_code_dict[code])
#         # df.at[index, 'ICD codes'] = new_codes
#     return df, len(new_code_dict)

def find_unique_code(df):
    """ 
    Find all unique codes in df, and returns the number of unique codes
    """
    all_codes = list(df['ICD codes'])
    all_codes = [item for sublist in all_codes for item in sublist]
    unique_codes = list(set(all_codes))
    unique_code_dict = {}
    for index, code in enumerate(unique_codes):
        unique_code_dict[code] = index
    return unique_code_dict, len(unique_codes) 
    


# def plot_code_distn(df):
#     """ 
#     Plot code distribution for males and females, (codes are updated)

#     We expect the distributions for males and females are different

#     Note: we prefer not to use the updated codes directly afterwards. 

#     """

#     df, _ = update_codes(df)
#     female_1_df = df.loc[(df['label'] == 1) & (df['gender'] == 'F')]
#     male_1_df = df.loc[(df['label'] == 1) & (df['gender'] == 'M')]
#     female_0_df = df.loc[(df['label'] == 0) & (df['gender'] == 'F')]
#     male_0_df = df.loc[(df['label'] == 0) & (df['gender'] == 'M')]
#     print("female_1_df.shape is:", female_1_df.shape)
#     print("female_0_df.shape is:", female_0_df.shape)
#     print("male_1_df.shape is:", male_1_df.shape)
#     print("male_0_df.shape is:", male_0_df.shape)
#     bin_width = 1

#     plt.subplot(2,2,1)
#     female_1_codes = female_1_df['ICD codes']
#     female_1_codes = [code for sublist in female_1_codes for code in sublist]
#     plt.hist(female_1_codes, bins=range(min(female_1_codes), max(female_1_codes)+bin_width, bin_width))
#     plt.title("female label 1")

#     plt.subplot(2,2,2)
#     male_1_codes = male_1_df['ICD codes']
#     male_1_codes = [code for sublist in male_1_codes for code in sublist]
#     plt.hist(male_1_codes, bins=range(min(male_1_codes), max(male_1_codes)+bin_width, bin_width))
#     plt.title("male label 1")

#     plt.subplot(2,2,3)
#     female_0_codes = female_0_df['ICD codes']
#     female_0_codes = [code for sublist in female_0_codes for code in sublist]
#     plt.hist(female_0_codes, bins=range(min(female_0_codes), max(female_0_codes)+bin_width, bin_width))
#     plt.title("female label 0")

#     plt.subplot(2,2,4)
#     male_0_codes = male_0_df['ICD codes']
#     male_0_codes = [code for sublist in male_0_codes for code in sublist]
#     plt.hist(male_0_codes, bins=range(min(male_0_codes), max(male_0_codes)+bin_width, bin_width))
#     plt.title("male label 0")

#     plt.tight_layout()
#     plt.show()


def gen_features_labels(df, label_code):
    """ 
    Generate source features, source labels, target features and target labels from dataframe df

    TODO: Now we assume female is the source, and male is the target.

    """

    unique_code_dict, num_codes = find_unique_code(df)
    # print("number of unique code is:", num_codes)
    # print("label code in unique_code_dict is:", unique_code_dict[label_code])

    source_df = df.loc[df['gender'] == 'M']
    target_df = df.loc[df['gender'] == 'F']

    # Prepare source
    source_features = np.empty(shape=[source_df.shape[0], num_codes])
    feature_index = 0
    for _, row in source_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row["ICD codes"]:
            if code != label_code:
                code_ind[unique_code_dict[code]] += 1
        source_features[feature_index] = code_ind
        feature_index += 1
    source_labels = np.array(list(source_df['label']))

    # Prepare target
    target_features = np.empty(shape=[target_df.shape[0], num_codes])
    feature_index = 0
    for _, row in target_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row["ICD codes"]:
            if code != label_code:
                code_ind[unique_code_dict[code]] += 1
        target_features[feature_index] = code_ind
        feature_index += 1
    target_labels = np.array(list(target_df['label']))

    return source_features, source_labels, target_features, target_labels


def gen_features_duration(df, group_name, group_1, group_2):
    """ 
    Generate source features, source durations (continuous response), \
        target features and target durations (continuous response) from dataframe df
    "duration" is the common response word for both experiment 4 and experiment 5

    """

    unique_code_dict, num_codes = find_unique_code(df)

    source_df = df.loc[df[group_name] == group_1]
    target_df = df.loc[df[group_name] == group_2]

    # Prepare source
    source_features = np.empty(shape=[source_df.shape[0], num_codes])
    feature_index = 0
    for _, row in source_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row["ICD codes"]:
            code_ind[unique_code_dict[code]] += 1
        source_features[feature_index] = code_ind
        feature_index += 1
    source_durations = np.array(list(source_df['duration']))

    # Prepare target
    target_features = np.empty(shape=[target_df.shape[0], num_codes])
    feature_index = 0
    for _, row in target_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row["ICD codes"]:
            code_ind[unique_code_dict[code]] += 1
        target_features[feature_index] = code_ind
        feature_index += 1
    target_durations = np.array(list(target_df['duration']))

    return source_features, source_durations, target_features, target_durations


def select_df_cts(df, group_name, group_1, group_2, group_1_count, group_2_count):
    """ 
    Select row in the dataframe df with balanced number of labels for males and females \
        for binary reponse

    :param Dataframe df: the dataframe to select samples from.
    :param int target_count: the number of samples of males.
    :param int source_count: the number of samples of females.

    :returns:
        - the selected dataframe
    """

    # generate label column based on label_code
    df_copy = copy.deepcopy(df)
    
    group_1_indices = []
    group_2_indices = []
    other_indices = []
    for index, row in df_copy.iterrows():
        if row[group_name] == group_1:
            group_1_indices.append(index)
        elif row[group_name] == group_2:
            group_2_indices.append(index)
        else:
            other_indices.append(index)

    
    # indices to delete from the dataframe
    # sample the same number of label 0s and label 1s

    delete_group_2_indices = random.sample(group_2_indices, len(group_2_indices)-group_2_count)
    delete_group_1_indices = random.sample(group_1_indices, len(group_1_indices)-group_1_count)

    delete_group_2_indices.extend(delete_group_1_indices)
    delete_group_2_indices.extend(other_indices)

    df_copy = df_copy.drop(delete_group_2_indices, axis=0, inplace=False)

    return df_copy


def select_df_binary(df, group_name, group_1, group_2, label_code, male_count, female_count):
    """ 
    Select row in the dataframe df with balanced number of labels for males and females
    Specifically, we want to reduce the number of rows with label 0 for males and females

    :param Dataframe df: the dataframe to select samples with label 0 and label 1
    :param str group_name: the criteria for dividing groups, e.g., "gender", "adm_type"
    :param str group_1: group 1 label for group_name
    :param str group_2: group 2 label for group_name
    :param str label_code: the ICD code for determining labels. This code should be removed from ICD codes.
    :param int target_count: the number of samples with label 1s and label 0s for target (male). 
    :param int source_count: the number of samples with label 1s and label 0s for source (female). 

    :returns:
        - the selected dataframe
    """
    print(f"label_code is {label_code}")
    # print(f"label_code is {label_code}, group 1 is {group_1}, group 2 is {group_2}")

    # select samples based on counts
    female_1_indices = []
    female_0_indices = []
    male_1_indices = []
    male_0_indices = []

    # generate label column based on label_code
    df_copy = copy.deepcopy(df)
    if 'label' in df_copy.columns:
        df_copy = df_copy.drop(['label'], axis=1)
    labels = []
    for index, row in df_copy.iterrows():
        if label_code in row['ICD codes']:
            labels.append(1)
        else:
            labels.append(0)
    df_copy['label'] = labels

    for index, row in df_copy.iterrows():
        if row['label'] == 0 and row[group_name] == group_1:
            female_0_indices.append(index)
        elif row['label'] == 0 and row[group_name] == group_2:
            male_0_indices.append(index)
        elif row['label'] == 1 and row[group_name] == group_1:
            female_1_indices.append(index)
        elif row['label'] == 1 and row[group_name] == group_2:
            male_1_indices.append(index)
    print("group 1 with label 0 length is:", len(female_0_indices))
    print("group 2 with label 0 length is:", len(male_0_indices))
    print("group 1 with label 1 length is:", len(female_1_indices))
    print("group 2 with label 1 length is:", len(male_1_indices))
    
    # indices to delete from the dataframe
    # sample the same number of label 0s and label 1s
    delete_female_0_indices = random.sample(female_0_indices, len(female_0_indices)-female_count)
    delete_male_0_indices = random.sample(male_0_indices, len(male_0_indices)-male_count)
    
    delete_female_1_indices = random.sample(female_1_indices, len(female_1_indices)-female_count)
    delete_male_1_indices = random.sample(male_1_indices, len(male_1_indices)-male_count)

    delete_female_0_indices.extend(delete_male_0_indices)
    delete_female_0_indices.extend(delete_female_1_indices)
    delete_female_0_indices.extend(delete_male_1_indices)
    
    df_copy = df_copy.drop(delete_female_0_indices, axis=0, inplace=False)
    
    return df_copy


def train_model(reps, labels, model_func):
    """ 
    Trains a model using reps and labels and returns the model
    """
    clf = model_func()
    clf.fit(reps, labels)
    return clf


def compute_transfer_score(source_reps, source_labels, target_reps, target_labels, model_func):
    """ 
    Computes the transfer score using the third term in Theorem 1: \mathbb{E}_{x\sim D_T}[|f_T(x)-f_S(x)|]
    where f_T is the labeling function for target domain, f_S is the labeling function for the source domain

    :param function model_func: the function to model the relationship between transported source reps and source labels
    """

    source_model = train_model(source_reps, source_labels, model_func)
    source_pred_probs = source_model.predict_proba(target_reps)

    target_model = train_model(target_reps, target_labels, model_func)
    target_pred_probs = target_model.predict_proba(target_reps)

    diff_probs = np.subtract(source_pred_probs, target_pred_probs)
    diff_probs = [abs(prob) for prob in diff_probs]

    return np.sum(diff_probs)



def entire_proc_binary(n_components, group_name, group_1, group_2, label_code, full_df, custom_train_reps, model_func, trans_metric, \
                       male_count = 120, female_count = 100, pca_explain=False, transfer_score=False, max_iter = None):
    """
    Wrap up the entire procedure

    :param int n_components: the number of components for PCA learning
    :param str group_name: group name to divide groups
    :param str group_1: group 1 name, the target in transfer learning (e.g. female)
    :param str group_2: group 2 name, the source in transfer learning (e.g. male)
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param function model_func: the function to model the relationship between target reps and target labels
    :param str trans_metric: transporting metric, NN, TCA, MMD or OT
    :param int male_count: the number of samples with label 1s and label 0s for target (male)
    :param int female_count: the number of samples with label 1s and label 0s for source (female)
    :param bool pca_explain: print the variance explained by the PCA, if True. Default False
    :param bool transfer_score: whether to compute transferability score. Default False. Returns the scores if True.
    :param int max_iter: maximum number of iteration for OT

    :returns accuracy statistics and optimized Wasserstein distance
    """
    
    selected_df = select_df_binary(full_df, group_name, group_1, group_2, \
                label_code, male_count, female_count)

    source_features, source_labels, target_features, target_labels = gen_features_labels(selected_df, label_code)

    source_reps = None
    target_reps = None
    if trans_metric != 'TCA':
        source_reps, target_reps = custom_train_reps(source_features, target_features, n_components, pca_explain=pca_explain)

    wa_dist = None

    if trans_metric == 'OT':
        trans_target_reps, wa_dist = trans_target2source(target_reps, source_reps, ret_cost=True, max_iter=max_iter)
    elif trans_metric == 'MMD':
        trans_target_reps = trans_MMD(target_reps, source_reps)
    elif trans_metric == 'TCA':
        source_reps, target_reps, trans_target_reps = TCA(source_features, target_features, n_components=n_components)
    
    source_model = train_model(source_reps, source_labels, model_func)
    source_preds = source_model.predict(source_reps)
    target_preds = source_model.predict(target_reps)
    trans_target_preds = source_model.predict(trans_target_reps)

    # Compute accuracies
    source_accuracy = accuracy_score(source_labels, source_preds)
    source_precision = precision_score(source_labels, source_preds)
    source_recall = recall_score(source_labels, source_preds)
    source_f1 = f1_score(source_labels, source_preds)

    target_accuracy = accuracy_score(target_labels, target_preds)
    target_precision = precision_score(target_labels, target_preds)
    target_recall = recall_score(target_labels, target_preds)
    target_f1 = f1_score(target_labels, target_preds)

    trans_target_accuracy = accuracy_score(target_labels, trans_target_preds)
    trans_target_precision = precision_score(target_labels, trans_target_preds)
    trans_target_recall = recall_score(target_labels, trans_target_preds)
    trans_target_f1 = f1_score(target_labels, trans_target_preds)
        
    if trans_metric == 'OT' and transfer_score:
        transfer_score = compute_transfer_score(source_reps, source_labels, target_reps, target_labels, model_func)
        return source_accuracy, source_precision, source_recall, source_f1, \
            target_accuracy, target_precision, target_recall, target_f1, \
            trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1,\
            transfer_score, wa_dist

    return source_accuracy, source_precision, source_recall, source_f1, \
        target_accuracy, target_precision, target_recall, target_f1, \
        trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1


def entire_proc_nn(n_components, group_name, group_1, group_2, label_code, full_df, custom_train_reps, type, model_func=linear_model.LogisticRegression, \
                       male_count = 120, female_count = 100, pca_explain=False, max_iter = None):
    """
    Wrap up the entire procedure using nearest neighbors

    :param int n_components: the number of components for PCA learning
    :param str group_name: group name to divide groups
    :param str group_1: group 1 name, the target in transfer learning (e.g. female)
    :param str group_2: group 2 name, the source in transfer learning (e.g. male)
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param str type: the task type, either classification or regression
    :param function model_func: the function to model the relationship between target reps and target labels
    :param int male_count: the number of samples with label 1s and label 0s for target (male)
    :param int female_count: the number of samples with label 1s and label 0s for source (female)
    :param bool pca_explain: print the variance explained by the PCA, if True. Default False
    :param bool transfer_score: whether to compute transferability score. Default False. Returns the scores if True.
    :param int max_iter: maximum number of iteration for OT

    :returns accuracy statistics and optimized Wasserstein distance
    """
    
    selected_df = select_df_binary(full_df, group_name, group_1, group_2, \
                label_code, male_count, female_count)

    source_features, source_labels, target_features, target_labels = gen_features_labels(selected_df, label_code)

    source_reps, target_reps = custom_train_reps(source_features, target_features, n_components, pca_explain=pca_explain)

    source_model = train_model(source_reps, source_labels, model_func)
    source_preds = source_model.predict(source_reps)
    target_preds = source_model.predict(target_reps)
    trans_target_reps = trans_NN(target_reps, source_reps, ret_cost=True, max_iter=max_iter)
    trans_target_preds = None
    if type == 'regression':
        trans_target_preds = source_model.predict(trans_target_reps)
    elif type == 'classification':
        for trans_target_rep in trans_target_reps:
            trans_target_preds_one_sample = source_model.predict(trans_target_rep7)


    # Compute accuracies
    source_accuracy = accuracy_score(source_labels, source_preds)
    source_precision = precision_score(source_labels, source_preds)
    source_recall = recall_score(source_labels, source_preds)
    source_f1 = f1_score(source_labels, source_preds)

    target_accuracy = accuracy_score(target_labels, target_preds)
    target_precision = precision_score(target_labels, target_preds)
    target_recall = recall_score(target_labels, target_preds)
    target_f1 = f1_score(target_labels, target_preds)

    trans_target_accuracy = accuracy_score(target_labels, trans_target_preds)
    trans_target_precision = precision_score(target_labels, trans_target_preds)
    trans_target_recall = recall_score(target_labels, trans_target_preds)
    trans_target_f1 = f1_score(target_labels, trans_target_preds)
        

    return source_accuracy, source_precision, source_recall, source_f1, \
        target_accuracy, target_precision, target_recall, target_f1, \
        trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1


def entire_proc_cts(n_components, full_df, custom_train_reps, model_func, trans_metric, \
                    group_name, group_1, group_2, group_1_count = 120, \
                    group_2_count = 100, pca_explain=False, equity=False, suffix=None):
    """
    Wrap up the entire procedure

    :param int n_components: the number of components for PCA learning
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param function model_func: the function the model the relationship between target representations and target response
    :param str trans_metric: transport metric, OT, MMD, TCA or NN
    :param str group_name: the group name 
    :param int male_count: the number of samples with label 1s and label 0s for target (male). Default 120.
    :param int female_count: the number of samples with label 1s and label 0s for source (female). Default 100.
    :param bool pca_explain: print the variance explained by the PCA, if True. Default False.
    :param bool equity: track differences in predicted values versus ground-truth values, to prepare data to check for equity
    """
    
    selected_df = select_df_cts(full_df, group_name, group_1, group_2, group_1_count=group_1_count, group_2_count=group_2_count)

    source_features, source_labels, target_features, target_labels = gen_features_duration(selected_df, group_name, group_1, group_2)

    source_reps = None
    target_reps = None
    if trans_metric != 'TCA':
        source_reps, target_reps = custom_train_reps(source_features, target_features, n_components, pca_explain=pca_explain)

    trans_target_reps = None
    coupling = None
    if trans_metric == 'OT':
        trans_target_reps, coupling = trans_target2source(target_reps, source_reps, max_iter=10000000, ret_coupling=True)
    if trans_metric == 'MMD':
        trans_target_reps = trans_MMD(target_reps, source_reps)
    elif trans_metric == 'TCA':
        source_reps, target_reps, trans_target_reps = TCA(source_features, target_features, n_components=n_components)

    clf = train_model(source_reps, source_labels, model_func) 
    source_preds = clf.predict(source_reps)
    target_preds = clf.predict(target_reps)
    trans_target_preds = clf.predict(trans_target_reps)

    if equity and trans_metric == 'OT':
        # compute transported target without using the model, used for studying the bias 
        trans_target_mappings = np.matmul(coupling, source_labels)
        trans_target_mappings = np.multiply(trans_target_mappings, group_2_count)
        target_equity_path = os.path.join(mimic_output_dir, f"exp4_{group_name}_{group_2}2{group_1}_equity.csv")
        if suffix is not None:
            target_equity_path = os.path.join(mimic_output_dir, f"exp4_{group_name}_{group_2}2{group_1}_{suffix}_equity.csv")
        target_equity_df = pd.read_csv(target_equity_path, header=0, index_col=None)
        target_diffs = np.divide(trans_target_mappings - target_labels, target_labels)

        target_data_block = np.transpose(np.array([target_labels, trans_target_mappings, target_diffs]))
        target_new_df = pd.DataFrame(target_data_block, columns=target_equity_df.columns)
        target_equity_df = pd.concat([target_equity_df, target_new_df], ignore_index=True)
        target_equity_df.to_csv(target_equity_path, index=False, header=True)


    source_mae = metrics.mean_absolute_error(source_labels, source_preds)
    source_mse = mean_squared_error(source_labels, source_preds)
    source_rmse = np.sqrt(metrics.mean_squared_error(source_labels, source_preds))
    target_mae = metrics.mean_absolute_error(target_labels, target_preds)
    target_mse = mean_squared_error(target_labels, target_preds)
    target_rmse = np.sqrt(metrics.mean_squared_error(target_labels, target_preds))
    trans_target_mae = metrics.mean_absolute_error(target_labels, trans_target_preds)
    trans_target_mse = mean_squared_error(target_labels, trans_target_preds)
    trans_target_rmse = np.sqrt(metrics.mean_squared_error(target_labels, trans_target_preds))

    return source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse, \
        trans_target_mae, trans_target_mse, trans_target_rmse


def multi_proc_binary(score_path, n_components, label_code, full_df, custom_train_reps, \
               male_count, female_count, iteration=20, max_iter = None):
    """ 
    Run the entire_proc function multiple times (iteration) for binary responses

    :param str score_path: the path to save results
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param int n_components: the number of components in PCA to learn representations
    :param int male_count: the number of samples with label 1s and label 0s for target (male)
    :param int female_count: the number of samples with label 1s and label 0s for source (female)
    :param int iteration: the number of iterations (repetitions)
    :param int max_iter: maximum number of iterations for OT
    """

    random.seed(0)
    res = np.empty(shape=[iteration, 12])
    for i in range(iteration):
        print("iteration:", i)
        cur_res = entire_proc_binary(n_components, label_code, full_df, custom_train_reps, male_count, female_count, max_iter=max_iter)
        res[i] = cur_res
    res_df = pd.DataFrame(res, \
                          columns = ['source_accuracy', 'source_precision', 'source_recall', 'source_f1', \
                                     'target_accuracy', 'target_precision', 'target_recall', 'target_f1', \
                                        'trans_target_accuracy', 'trans_target_precision', 'trans_target_recall', 'trans_target_f1'])
    res_df.to_csv(score_path, index=False, header=True)
    return res



def entire_proc_binary_tca(n_components, group_name, group_1, group_2, label_code, full_df, custom_train_reps, model_func=linear_model.LogisticRegression, \
                       male_count = 120, female_count = 100):
    """ 
    Run the entire procedure on unordered response using TCA

    :param DataFrame df: the complete dataframe to select males and females from
    """
    selected_df = select_df_binary(full_df, group_name, group_1, group_2, \
                label_code, male_count, female_count)
    
    source_features, source_labels, target_features, target_labels = gen_features_labels(selected_df, label_code)
    # source_embs, target_embs = custom_train_reps(source_features, target_features, n_components)
    trans_source_embs, target_embs, trans_target_embs = TCA(source_features, target_features, n_components=n_components, scale=False)
    clf = model_func()
    clf.fit(trans_source_embs, source_labels)
    trans_target_preds = clf.predict(trans_target_embs)
    target_preds = clf.predict(target_embs)
    trans_source_preds = clf.predict(trans_source_embs)

    # Compute accuracies
    source_accuracy = accuracy_score(source_labels, trans_source_preds)
    source_precision = precision_score(source_labels, trans_source_preds)
    source_recall = recall_score(source_labels, trans_source_preds)
    source_f1 = f1_score(source_labels, trans_source_preds)

    target_accuracy = accuracy_score(target_labels, target_preds)
    target_precision = precision_score(target_labels, target_preds)
    target_recall = recall_score(target_labels, target_preds)
    target_f1 = f1_score(target_labels, target_preds)

    trans_target_accuracy = accuracy_score(target_labels, trans_target_preds)
    trans_target_precision = precision_score(target_labels, trans_target_preds)
    trans_target_recall = recall_score(target_labels, trans_target_preds)
    trans_target_f1 = f1_score(target_labels, trans_target_preds)

    # print("report f1:", target_f1, trans_target_f1)

    return source_accuracy, source_precision, source_recall, source_f1, \
        target_accuracy, target_precision, target_recall, target_f1, \
        trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1



def multi_proc_cts(n_components, full_df, custom_train_reps, \
               group_name, group_1, group_2, group_1_count, group_2_count, \
                trans_metric, model_func = linear_model.LinearRegression, iteration=20, equity=False, suffix=None):
    """ 
    Run the entire_proc function multiple times (iteration) for continuous responses

    :param str score_path: the path to save results
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param int n_components: the number of components in PCA to learn representations
    :param str group_name: the name of the dividing group
    :param str group_1: value for group 1
    :param str group_2: value for group 2
    :param int group_1_count: the number of samples with label 1s and label 0s for target (male)
    :param int group_2_count: the number of samples with label 1s and label 0s for source (female)
    :param str trans_metric: transport metric, OT, MMD, TCA or NN
    :param int iteration: the number of iterations (repetitions)
    :param bool equity: prepare statistics for equity
    """
    random.seed(0)
    source_maes = []
    source_mses = []
    source_rmses = [] 
    target_maes = []
    target_mses = [] 
    target_rmses = [] 
    trans_target_maes = []
    trans_target_mses = []
    trans_target_rmses = []

    if equity:
        target_equity_df = pd.DataFrame(columns=['target_label', 'target_pred_label', 'target_diff_percent'])
        target_equity_path = os.path.join(mimic_output_dir, f"exp4_{group_name}_{group_2}2{group_1}_equity.csv")
        if suffix is not None:
            target_equity_path = os.path.join(mimic_output_dir, f"exp4_{group_name}_{group_2}2{group_1}_{suffix}_equity.csv")
        target_equity_df.to_csv(target_equity_path, header=True, index=False)

    for i in range(iteration):
        print("iteration:", i)
        source_mae = None
        source_mse = None
        source_rmse = None 
        target_mae = None
        target_mse = None
        target_rmse = None 
        trans_target_mae = None
        trans_target_mse = None
        trans_target_rmse = None

        source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse, \
            trans_target_mae, trans_target_mse, trans_target_rmse = \
                entire_proc_cts(n_components, full_df, custom_train_reps, model_func, trans_metric, \
                    group_name, group_1, group_2, group_1_count, group_2_count, equity=equity, suffix=suffix)
        
        source_maes.append(source_mae)
        source_mses.append(source_mse)
        source_rmses.append(source_rmse)
        target_maes.append(target_mae)
        target_mses.append(target_mse)
        target_rmses.append(target_rmse)
        trans_target_maes.append(trans_target_mae)
        trans_target_mses.append(trans_target_mse)
        trans_target_rmses.append(trans_target_rmse)

    return source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
        trans_target_maes, trans_target_mses, trans_target_rmses


def multi_proc_binary_tca(n_components, full_df, custom_train_reps, \
               male_count, female_count, model_func = linear_model.LinearRegression, iteration=20):
    """ 
    Run the entire_proc function multiple times (iteration) for continuous responses

    :param str score_path: the path to save results
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param int n_components: the number of components in PCA to learn representations
    :param int male_count: the number of samples with label 1s and label 0s for target (male)
    :param int female_count: the number of samples with label 1s and label 0s for source (female)
    :param int iteration: the number of iterations (repetitions)
    """
    random.seed(0)
    source_accuracies = []
    source_precisions = []
    source_recalls = [] 
    source_f1s = []
    target_accuracies = []
    target_precisions = []
    target_recalls = [] 
    target_f1s = []
    trans_target_accuracies = []
    trans_target_precisions = []
    trans_target_recalls = [] 
    trans_target_f1s = []
    for i in range(iteration):
        print("iteration:", i)

        source_accuracy, source_precision, source_recall, source_f1, \
        target_accuracy, target_precision, target_recall, target_f1, \
        trans_target_accuracy, trans_target_precision, trans_target_recall, trans_target_f1 \
            = entire_proc_binary_tca(n_components, full_df, custom_train_reps, model_func, male_count=male_count, female_count=female_count)
        
        source_accuracies.append(source_accuracy)
        source_precisions.append(source_precision)
        source_recalls.append(source_recall)
        source_f1s.append(source_f1)
        target_accuracies.append(target_accuracy)
        target_precisions.append(target_precision)
        target_recalls.append(target_recall)
        target_f1s.append(target_f1)
        trans_target_accuracies.append(trans_target_accuracy)
        trans_target_precisions.append(trans_target_precision)
        trans_target_recalls.append(trans_target_recall)
        trans_target_f1s.append(trans_target_f1)

    return source_accuracies, source_precisions, source_recalls, source_f1s, \
        target_accuracies, target_precisions, target_recalls, target_f1s, \
        trans_target_accuracies, trans_target_precisions, trans_target_recalls, trans_target_f1s


def build_maps(admission_file, diagnosis_file, patient_file):
    """ 
    Building pid-admission mapping, pid-gender mapping, admission-date mapping, admission-code mapping and pid-sorted visits mapping
    :param str admission_file: path to ADMISSIONS.csv
    :param str diagnosis_file: path to DIAGNOSES_ICD.csv
    :param str patient_file: path to PATIENTS.csv
    """

    # Building pid-admissions mapping and pid-date mapping
    pid_adms = {}
    adm_date = {}
    infd = open(admission_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admid = int(tokens[2])
        adm_time = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        adm_date[admid] = adm_time
        if pid in pid_adms: pid_adms[pid].append(admid)
        else: pid_adms[pid] = [admid]
    infd.close()

    # Bulding admission-codes mapping
    admid_codes = {}
    infd = open(diagnosis_file, 'r')
    infd.readline()
    for line in infd: # read ADMISSIONS.CSV in order
        tokens = line.strip().split(',')
        admid = int(tokens[2])
        code = tokens[4][1:-1]

        if admid in admid_codes: 
            admid_codes[admid].append(code)
        else: 
            admid_codes[admid] = [code]
    infd.close()
    
    # Building pid-sorted visits mapping, a visit consists of (time, codes)
    pid_visits = {}
    for pid, adms in pid_adms.items():
        if len(adms) < 2: continue

        # sort by date
        sorted_visit = sorted([(adm_date[admid], admid_codes[admid]) for admid in adms])
        pid_visits[pid] = sorted_visit
    
    pid_gender = {}
    infd = open(patient_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        gender = str(tokens[2])
        pid_gender[pid] = gender[1] # remove quotes
    infd.close()
                
    return pid_adms, pid_gender, adm_date, admid_codes, pid_visits


def entire_proc_cts_tca(df, male_count, female_count, model_func, n_components):
    """ 
    Run the entire procedure on ordered response using TCA

    :param DataFrame df: the complete dataframe to select males and females from
    """
    selected_df = select_df_cts(df, male_count, female_count)
    source_features, source_labels, target_features, target_labels = gen_features_duration(selected_df)
    trans_source_embs, target_embs, trans_target_embs = \
        TCA(source_features, target_features, n_components=n_components)
    clf = model_func()
    clf.fit(trans_source_embs, source_labels)
    trans_target_pred = clf.predict(trans_target_embs)
    target_pred = clf.predict(target_embs)
    trans_source_pred = clf.predict(trans_source_embs)
    source_mae = mean_absolute_error(source_labels, trans_source_pred)
    source_mse = mean_squared_error(source_labels, trans_source_pred)
    source_rmse = np.sqrt(mean_squared_error(source_labels, trans_source_pred))
    target_mae = mean_absolute_error(target_labels, target_pred)
    target_mse = mean_squared_error(target_labels, target_pred)
    target_rmse = np.sqrt(mean_squared_error(target_labels, target_pred))
    trans_target_mae = mean_absolute_error(target_labels, trans_target_pred)
    trans_target_mse = mean_squared_error(target_labels, trans_target_pred)
    trans_target_rmse = np.sqrt(mean_squared_error(target_labels, trans_target_pred))
    return source_mae, source_mse, source_rmse, \
        target_mae, target_mse, target_rmse, \
        trans_target_mae, trans_target_mse, trans_target_rmse
        

def multi_proc_cts_tca(df, model_func, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for continuous labels using TCA

    :param DataFrame df: the complete dataframe to select males and females from
    :param function model_func: the function to model the relationship between representations and responses

    :returns: vectors of accuracy statistics of multiple rounds
    """
    random.seed(0)
    source_maes = []
    source_mses = []
    source_rmses = [] 
    target_maes = []
    target_mses = [] 
    target_rmses = [] 
    trans_target_maes = []
    trans_target_mses = []
    trans_target_rmses = []

    for i in range(n_times):
        print("iteration:", i)
        # init accuracies
        source_mae = None
        source_mse = None
        source_rmse = None 
        target_mae = None
        target_mse = None
        target_rmse = None 
        trans_target_mae = None
        trans_target_mse = None
        trans_target_rmse = None


        n_components = 50
        male_count = 120
        female_count = 100
        source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse,\
            trans_target_mae, trans_target_mse, trans_target_rmse = \
                entire_proc_cts_tca(df, male_count, female_count, model_func, n_components)
    
        source_maes.append(source_mae)
        source_mses.append(source_mse)
        source_rmses.append(source_rmse)
        target_maes.append(target_mae)
        target_mses.append(target_mse)
        target_rmses.append(target_rmse)
        trans_target_maes.append(trans_target_mae)
        trans_target_mses.append(trans_target_mse)
        trans_target_rmses.append(trans_target_rmse)
    return source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
        trans_target_maes, trans_target_mses, trans_target_rmses


def entire_proc_cts_NN(df, custom_train_reps, n_neighbors, male_count, female_count, model_func, n_components):
    """ 
    Run the entire procedure on ordered response using NN

    :param DataFrame df: the complete dataframe to select males and females from
    """
    selected_df = select_df_cts(df, male_count, female_count)
    source_features, source_labels, target_features, target_labels = gen_features_duration(selected_df)
    source_embs, target_embs = custom_train_reps(source_features, target_features, n_components)
    trans_target_embs = trans_NN(source_embs, target_embs, n_neighbors, 'regression')
    clf = model_func()
    clf.fit(source_embs, source_labels)
    trans_target_pred = clf.predict(trans_target_embs)
    target_pred = clf.predict(target_embs)
    source_pred = clf.predict(source_embs)
    source_mae = mean_absolute_error(source_labels, source_pred)
    source_mse = mean_squared_error(source_labels, source_pred)
    source_rmse = np.sqrt(mean_squared_error(source_labels, source_pred))
    target_mae = mean_absolute_error(target_labels, target_pred)
    target_mse = mean_squared_error(target_labels, target_pred)
    target_rmse = np.sqrt(mean_squared_error(target_labels, target_pred))
    trans_target_mae = mean_absolute_error(target_labels, trans_target_pred)
    trans_target_mse = mean_squared_error(target_labels, trans_target_pred)
    trans_target_rmse = np.sqrt(mean_squared_error(target_labels, trans_target_pred))
    return source_mae, source_mse, source_rmse, \
        target_mae, target_mse, target_rmse, \
        trans_target_mae, trans_target_mse, trans_target_rmse


def multi_proc_cts_NN(df, model_func, n_neighbors, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
    for continuous labels using nearest neighbors 

    :param DataFrame df: the complete dataframe to select males and females from
    :param function model_func: the function to model the relationship between representations and responses

    :returns: vectors of accuracy statistics of multiple rounds
    """
    random.seed(0)
    source_maes = []
    source_mses = []
    source_rmses = [] 
    target_maes = []
    target_mses = [] 
    target_rmses = [] 
    trans_target_maes = []
    trans_target_mses = []
    trans_target_rmses = []

    for i in range(n_times):
        print("iteration:", i)
        # init accuracies
        source_mae = None
        source_mse = None
        source_rmse = None 
        target_mae = None
        target_mse = None
        target_rmse = None 
        trans_target_mae = None
        trans_target_mse = None
        trans_target_rmse = None


        n_components = 50
        male_count = 120
        female_count = 100
        source_mae, source_mse, source_rmse, target_mae, target_mse, target_rmse,\
            trans_target_mae, trans_target_mse, trans_target_rmse = \
                entire_proc_cts_NN(df, custom_train_reps_default, n_neighbors, male_count, female_count, model_func, n_components)
    
        source_maes.append(source_mae)
        source_mses.append(source_mse)
        source_rmses.append(source_rmse)
        target_maes.append(target_mae)
        target_mses.append(target_mse)
        target_rmses.append(target_rmse)
        trans_target_maes.append(trans_target_mae)
        trans_target_mses.append(trans_target_mse)
        trans_target_rmses.append(trans_target_rmse)
    return source_maes, source_mses, source_rmses, target_maes, target_mses, target_rmses,\
        trans_target_maes, trans_target_mses, trans_target_rmses


def custom_train_reps_default(source_features, target_features, n_components, pca_explain=False):
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


def decide_ICD_chapter(code):
    """ 
    Decide ICD code chapter based on 
    https://en.wikipedia.org/wiki/List_of_ICD-9_codes_E_and_V_codes:_external_causes_of_injury_and_supplemental_classification 

    Rules:
    Chapter	Block	Title
    1	001–139	Infectious and Parasitic Diseases
    2	140–239	Neoplasms
    3	240–279	Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders
    4	280–289	Diseases of the Blood and Blood-forming Organs
    5	290–319	Mental Disorders
    6	320–389	Diseases of the Nervous System and Sense Organs
    7	390–459	Diseases of the Circulatory System
    8	460–519	Diseases of the Respiratory System
    9	520–579	Diseases of the Digestive System
    10	580–629	Diseases of the Genitourinary System
    11	630–679	Complications of Pregnancy, Childbirth, and the Puerperium
    12	680–709	Diseases of the Skin and Subcutaneous Tissue
    13	710–739	Diseases of the Musculoskeletal System and Connective Tissue
    14	740–759	Congenital Anomalies
    15	760–779	Certain Conditions originating in the Perinatal Period
    16	780–799	Symptoms, Signs and Ill-defined Conditions
    17	800–999	Injury and Poisoning
    18  E800–E999   Supplementary Classification of External Causes of Injury and Poisoning
    19  V01–V82	Supplementary Classification of Factors influencing Health Status and Contact with Health Services
    20  M8000–M9970	Morphology of Neoplasms
    """

    block = code.split(".")[0]
    if block.startswith("E"):
        return 18, "Supplementary Classification of External Causes of Injury and Poisoning"
    if block.startswith("V"):
        return 19, "Supplementary Classification of Factors influencing Health Status and Contact with Health Services"
    if block.startswith("M"):
        return 20, "Morphology of Neoplasms"
    
    # otherwise, numerical code
    block_segs = [1, 140, 240, 280, 290, 320, 390, 460, 520, 580, 630, 680, 710, 740, 760, 780, 800, 1000]
    titles = ["Infectious and Parasitic Diseases", 
              "Neoplasms", 
              "Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders", \
        "Diseases of the Blood and Blood-forming Organs", 
        "Mental Disorders", 
        "Diseases of the Nervous System and Sense Organs", \
        "Diseases of the Circulatory System", 
        "Diseases of the Respiratory System", 
        "Diseases of the Digestive System", \
        "Diseases of the Genitourinary System", 
        "Complications of Pregnancy, Childbirth, and the Puerperium", 
        "Diseases of the Skin and Subcutaneous Tissue",\
        "Diseases of the Musculoskeletal System and Connective Tissue", 
        "Congenital Anomalies", 
        "Certain Conditions originating in the Perinatal Period", \
        "Symptoms, Signs and Ill-defined Conditions", 
        "Injury and Poisoning" ]
    block = int(block)
    for chapter_index in range(len(block_segs)):
        left = block_segs[chapter_index]
        right = block_segs[chapter_index+1]
        if left <= block and block < right:
            return chapter_index+1, titles[chapter_index]


def decide_all_ICD_chapters(codes):
    chapters = []
    for code in codes:
        chapter, _ = decide_ICD_chapter(code)
        chapters.append(chapter)
    return chapters


def get_label_codes():
    """ 
    Get label codes (the used label codes)
    """
    mimic_output_dir = "/home/wanxinli/EHR-OT/outputs/mimic"
    label_codes = []
    for file in os.listdir(mimic_output_dir):
        if file.endswith("OT_score.csv") and "exp3" in file:
            label_codes.append(file.split("_")[1])
    return label_codes


def compute_metric_ratio(score_df, eval_metric):
    """ 
    Computes the improvement ratios for a evaluation metric (eval_metric) on dataset (score_df). \
        Used for visualizing results.
    :param Dataframe score_df: the dataframe for scores
    :param str eval_metric: the metric name for computing ratios, can be mae or rmse for regression, \
        precision, recall and f1 for classification
    """
    improve_ratios = []
    for target_metric, trans_target_metric in zip(score_df[f'target_{eval_metric}'], score_df[f'trans_target_{eval_metric}']):
        if eval_metric == 'f1' or eval_metric == 'precision' or eval_metric == 'recall':
            improve_ratios.append(trans_target_metric-target_metric)
        elif eval_metric == 'mae' or eval_metric == 'rmse':
            improve_ratios.append((target_metric-trans_target_metric)/target_metric) # smaller is better
    return improve_ratios