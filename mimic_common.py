import sys
sys.path.append("/home/wanxinli/deep_patient/")

from datetime import datetime
import copy
from common import *
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
from multiprocess import Pool
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
    f1_score, mean_absolute_error, mean_squared_error, \
    mutual_info_score, normalized_mutual_info_score
from scipy.stats import entropy




def update_codes(df):
    """ 
    Update code in dataframe, the new code starts from 0.
    We use -1 to denote NA code in later analysis

    returns
        - a new dataframe with the updated codes
        - total number of unique codes in df
    """
    new_code_dict = {} # mapping from old code to new code
    for index, row in df.iterrows():
        cur_codes = row['ICD codes']
        new_codes = []
        for code in cur_codes:
            if code not in new_code_dict:
                new_code_dict[code] = len(new_code_dict) # add a new entry
            new_codes.append(new_code_dict[code])
        df.at[index, 'ICD codes'] = new_codes
    return df, len(new_code_dict)


def plot_code_distn(df):
    """ 
    Plot code distribution for males and females, (codes are updated)

    We expect the distributions for males and females are different

    Note: we prefer not to use the updated codes directly afterwards. 

    """

    df, _ = update_codes(df)
    female_1_df = df.loc[(df['label'] == 1) & (df['gender'] == 'F')]
    male_1_df = df.loc[(df['label'] == 1) & (df['gender'] == 'M')]
    female_0_df = df.loc[(df['label'] == 0) & (df['gender'] == 'F')]
    male_0_df = df.loc[(df['label'] == 0) & (df['gender'] == 'M')]
    print("female_1_df.shape is:", female_1_df.shape)
    print("female_0_df.shape is:", female_0_df.shape)
    print("male_1_df.shape is:", male_1_df.shape)
    print("male_0_df.shape is:", male_0_df.shape)
    bin_width = 1

    plt.subplot(2,2,1)
    female_1_codes = female_1_df['ICD codes']
    female_1_codes = [code for sublist in female_1_codes for code in sublist]
    plt.hist(female_1_codes, bins=range(min(female_1_codes), max(female_1_codes)+bin_width, bin_width))
    plt.title("female label 1")

    plt.subplot(2,2,2)
    male_1_codes = male_1_df['ICD codes']
    male_1_codes = [code for sublist in male_1_codes for code in sublist]
    plt.hist(male_1_codes, bins=range(min(male_1_codes), max(male_1_codes)+bin_width, bin_width))
    plt.title("male label 1")

    plt.subplot(2,2,3)
    female_0_codes = female_0_df['ICD codes']
    female_0_codes = [code for sublist in female_0_codes for code in sublist]
    plt.hist(female_0_codes, bins=range(min(female_0_codes), max(female_0_codes)+bin_width, bin_width))
    plt.title("female label 0")

    plt.subplot(2,2,4)
    male_0_codes = male_0_df['ICD codes']
    male_0_codes = [code for sublist in male_0_codes for code in sublist]
    plt.hist(male_0_codes, bins=range(min(male_0_codes), max(male_0_codes)+bin_width, bin_width))
    plt.title("male label 0")

    plt.tight_layout()
    plt.show()


def gen_features_labels(df):
    """ 
    Generate source features, source labels, target features and target labels from dataframe df

    TODO: Now we assume female is the source, and male is the target.

    """

    _, num_codes = update_codes(df)

    source_df = df.loc[df['gender'] == 'F']
    target_df = df.loc[df['gender'] == 'M']

    # Prepare target
    target_features = np.empty(shape=[target_df.shape[0], num_codes])
    feature_index = 0
    for _, row in target_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row["ICD codes"]:
            code_ind[code] += 1
        target_features[feature_index] = code_ind
        feature_index += 1
    target_labels = np.array(list(target_df['label']))

    # Prepare source
    source_features = np.empty(shape=[source_df.shape[0], num_codes])
    feature_index = 0
    for _, row in source_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row["ICD codes"]:
            code_ind[code] += 1
        source_features[feature_index] = code_ind
        feature_index += 1
    source_labels = np.array(list(source_df['label']))

    return target_features, target_labels, source_features, source_labels


def gen_features_duration(df):
    """ 
    Generate source features, source durations (continuous response), \
        target features and target durations (continuous response) from dataframe df

    TODO: Now we assume female is the source, and male is the target.

    """

    _, num_codes = update_codes(df)

    source_df = df.loc[df['gender'] == 'F']
    target_df = df.loc[df['gender'] == 'M']

    # Prepare target
    target_features = np.empty(shape=[target_df.shape[0], num_codes])
    feature_index = 0
    for _, row in target_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row["ICD codes"]:
            code_ind[code] += 1
        target_features[feature_index] = code_ind
        feature_index += 1
    target_durations = np.array(list(target_df['duration']))

    # Prepare source
    source_features = np.empty(shape=[source_df.shape[0], num_codes])
    feature_index = 0
    for _, row in source_df.iterrows():
        code_ind = np.zeros(num_codes)
        for code in row["ICD codes"]:
            code_ind[code] += 1
        source_features[feature_index] = code_ind
        feature_index += 1
    source_durations = np.array(list(source_df['duration']))

    return target_features, target_durations, source_features, source_durations


def select_df_cts(df, male_count, female_count):
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
    female_indices = []
    male_indices = []

    for index, row in df_copy.iterrows():
        if row['gender'] == 'F':
            female_indices.append(index)
        elif row['gender'] == 'M':
            male_indices.append(index)
    
    # indices to delete from the dataframe
    # sample the same number of label 0s and label 1s
    delete_female_indices = random.sample(female_indices, len(female_indices)-female_count)
    delete_male_indices = random.sample(male_indices, len(male_indices)-male_count)

    delete_female_indices.extend(delete_male_indices)
    
    df_copy = df_copy.drop(delete_female_indices, axis=0, inplace=False)

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
    # print("group 1 with label 0 length is:", len(female_0_indices))
    # print("group 2 with label 0 length is:", len(male_0_indices))
    # print("group 1 with label 1 length is:", len(female_1_indices))
    # print("group 2 with label 1 length is:", len(male_1_indices))
    
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

    # remove label_code from ICD code features
    for index, row in df_copy.iterrows():
        if label_code in row['ICD codes']:
            new_codes = copy.deepcopy(row['ICD codes'])
            new_codes.remove(label_code)
            df_copy.at[index, 'ICD codes'] = new_codes
    
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



def entire_proc_binary(n_components, group_name, group_1, group_2, label_code, full_df, custom_train_reps, model_func=linear_model.LogisticRegression, \
                       male_count = 120, female_count = 100, pca_explain=False, transfer_score=False):
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
    :param int male_count: the number of samples with label 1s and label 0s for target (male)
    :param int female_count: the number of samples with label 1s and label 0s for source (female)
    :param bool pca_explain: print the variance explained by the PCA, if True. Default False
    :param bool transfer_score: whether to compute transferability score. Default False. Returns the scores if True.

    :returns accuracy statistics and optimized Wasserstein distance
    """
    
    selected_df = select_df_binary(full_df, group_name, group_1, group_2, \
                label_code, male_count, female_count)

    target_features, target_labels, source_features, source_labels = gen_features_labels(selected_df)

    target_reps, source_reps = custom_train_reps(target_features, source_features, n_components, pca_explain=pca_explain)

    target_model = train_model(target_reps, target_labels, model_func)
    target_preds = target_model.predict(target_reps)
    source_preds = target_model.predict(source_reps)
    trans_source_reps, cost = trans_source2target(source_reps, target_reps, ret_cost=True)
    
    # TODO: get Wasserstein distance here 
    trans_source_preds = target_model.predict(trans_source_reps)
    target_accuracy = accuracy_score(target_labels, target_preds)
    target_precision = precision_score(target_labels, target_preds)
    target_recall = recall_score(target_labels, target_preds)
    target_f1 = f1_score(target_labels, target_preds)
    source_accuracy = accuracy_score(source_labels, source_preds)
    source_precision = precision_score(source_labels, source_preds)
    source_recall = recall_score(source_labels, source_preds)
    source_f1 = f1_score(source_labels, source_preds)
    trans_source_accuracy = accuracy_score(source_labels, trans_source_preds)
    trans_source_precision = precision_score(source_labels, trans_source_preds)
    trans_source_recall = recall_score(source_labels, trans_source_preds)
    trans_source_f1 = f1_score(source_labels, trans_source_preds)
        
    if transfer_score:
        transfer_score = compute_transfer_score(source_reps, source_labels, target_reps, target_labels, model_func)
        return target_accuracy, target_precision, target_recall, target_f1, \
            source_accuracy, source_precision, source_recall, source_f1, \
            trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1,\
            transfer_score, cost

    return target_accuracy, target_precision, target_recall, target_f1, \
        source_accuracy, source_precision, source_recall, source_f1, \
        trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1, cost


def entire_proc_cts(n_components, full_df, custom_train_reps, model_func, male_count = 120, female_count = 100, pca_explain=False):
    """
    Wrap up the entire procedure

    :param int n_components: the number of components for PCA learning
    :param str label_code: the ICD code to determine labels
    :param dataframe full_df: the full dataframe
    :param function custom_train_reps: the customized function for learning representations
    :param function model_func: the function the model the relationship between target representations and target response
    :param int male_count: the number of samples with label 1s and label 0s for target (male). Default 120.
    :param int female_count: the number of samples with label 1s and label 0s for source (female). Default 100.
    :param bool pca_explain: print the variance explained by the PCA, if True. Default False.
    """
    
    selected_df = select_df_cts(full_df, male_count=male_count, female_count=female_count)

    target_features, target_labels, source_features, source_labels = gen_features_duration(selected_df)

    target_reps, source_reps = custom_train_reps(target_features, source_features, n_components, pca_explain=pca_explain)

    clf = train_model(target_reps, target_labels, model_func) 
    target_preds = clf.predict(target_reps)
    source_preds = clf.predict(source_reps)
    trans_source_reps = trans_source2target(source_reps, target_reps)
    trans_source_preds = clf.predict(trans_source_reps)

    target_mae = metrics.mean_absolute_error(target_labels, target_preds)
    target_rmse = np.sqrt(metrics.mean_squared_error(target_labels, target_preds))
    source_mae = metrics.mean_absolute_error(source_labels, source_preds)
    source_rmse = np.sqrt(metrics.mean_squared_error(source_labels, source_preds))
    trans_source_mae = metrics.mean_absolute_error(source_labels, trans_source_preds)
    trans_source_rmse = np.sqrt(metrics.mean_squared_error(source_labels, trans_source_preds))

    return target_mae, target_rmse, source_mae, source_rmse, trans_source_mae, trans_source_rmse


def multi_proc_binary(score_path, n_components, label_code, full_df, custom_train_reps, \
               male_count, female_count, iteration=20):
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
    """
    res = np.empty(shape=[iteration, 12])
    for i in range(iteration):
        print("iteration:", i)
        cur_res = entire_proc_binary(n_components, label_code, full_df, custom_train_reps, male_count, female_count)
        res[i] = cur_res
    res_df = pd.DataFrame(res, \
                          columns = ['target_accuracy', 'target_precision', 'target_recall', 'target_f1', \
                                     'source_accuracy', 'source_precision', 'source_recall', 'source_f1', \
                                        'trans_source_accuracy', 'trans_source_precision', 'trans_source_recall', 'trans_source_f1'])
    res_df.to_csv(score_path, index=False, header=True)
    return res


def multi_proc_cts(score_path, n_components, full_df, custom_train_reps, \
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
    res = np.empty(shape=[iteration, 6])
    for i in range(iteration):
        print("iteration:", i)
        try:
            cur_res = entire_proc_cts(n_components, full_df, custom_train_reps, model_func, male_count=male_count, female_count=female_count)
            res[i] = cur_res
        except:
            print("cannot fit the regression model")
    res_df = pd.DataFrame(res, columns = ['target_mae', 'target_rmse', 'source_mae', 'source_rmse', 'trans_source_mae', 'trans_source_rmse'])
    res_df.to_csv(score_path, index=False, header=True)
    return res


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