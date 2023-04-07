""" 
Common functions for synthetic datasets
"""


import sys
sys.path.append("/home/wanxinli/deep_patient")

import numpy as np
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import ot
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mutual_info_score
from scipy.stats import wasserstein_distance

from statistics import median

""" 
Transport source representations to target representations
"""

def trans_source2target(source_reps, target_reps, type="balanced", max_iter = None, ret_cost=False):
    """ 
    Optimal transport (without entropy regularization) source representations \
        to target representations

    :param str type: balanced or unbalanced
    :param bool ret_cost: return OT cost or not
    :returns: transported source representations and the optimized Wasserstein distance (if cost is True), default False

    TODO: the unbalanced case has not been implemented 
    """
    trans_source_reps = None
    # if type == "balanced":
    ot_emd = ot.da.SinkhornTransport(reg_e=1e-1, log=True)
    if max_iter is not None:
        ot_emd = ot.da.SinkhornTransport(reg_e=1e-1, max_iter=max_iter, log=True)
    ot_emd.fit(Xs=source_reps, Xt=target_reps)
    trans_source_reps = ot_emd.transform(Xs=source_reps)
    if not ret_cost:
        return trans_source_reps
    wa_dist = ot_emd.log_['err'][-1]
    # wa_dist = wasserstein_distance(source_reps, target_reps)


    return trans_source_reps, wa_dist


    # elif type == "unbalanced":
    #     reg = 0.005
    #     reg_m_kl = 0.5
    #     n = source_reps.shape[0]

    #     a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

    #     M = ot.dist(source_reps, target_reps)
    #     M /= M.max()

    #     coupling = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m_kl)
    #     trans_source_reps = np.matmul(coupling, source_reps)



""" 
Caculate result statistics for binary labels
"""

def cal_stats_binary(target_reps, target_labels, source_reps, source_labels, \
    trans_source_reps, target_model):
    """ 
    Calculate accuracy statistics based on logistic regression between the \
        patient representations and label labels
    This function is for binary labels

    :param function target_model: the model trained by target data
    
    :returns: using the target model,\
        - accuracy for target/source/transported source
        - precision for target/source/transported source
        - recall for target/source/transported source
        - f1 for target/source/transported source
            
    """

    # calculate the stats
    target_pred_labels = target_model.predict(target_reps)
    target_accuracy = accuracy_score(target_labels, target_pred_labels)
    target_precision = precision_score(target_labels, target_pred_labels)
    target_recall = recall_score(target_labels, target_pred_labels)
    target_f1 = f1_score(target_labels, target_pred_labels, average="weighted")

    source_pred_labels = target_model.predict(source_reps)
    source_accuracy = accuracy_score(source_labels, source_pred_labels)
    source_precision = precision_score(source_labels, source_pred_labels)
    source_recall = recall_score(source_labels, source_pred_labels)
    source_f1 = f1_score(source_labels, source_pred_labels, average="weighted")

    trans_source_pred_labels = target_model.predict(trans_source_reps)
    trans_source_accuracy = accuracy_score(source_labels, trans_source_pred_labels)
    trans_source_precision = precision_score(source_labels, trans_source_pred_labels)
    trans_source_recall = recall_score(source_labels, trans_source_pred_labels)
    trans_source_f1 = f1_score(source_labels, trans_source_pred_labels, average="weighted")


    return target_accuracy, target_precision, target_recall, target_f1, \
        source_accuracy, source_precision, source_recall, source_f1, \
        trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1


def cal_stats_cts(target_reps, target_labels, source_reps, source_labels, \
    trans_source_reps, model_func):
    """ 
    Calculate accuracy statistics based on logistic regression between the \
        patient representations and label labels
    This function is for continous labels

    :param function model_func: the function to model the relationship between \
        representations and reponse
    
    :returns: using the target model,\
        - mean absoluate error (MAE) for target/source/transported source
        - mean squared error (MSE) for target/source/transported source
        - residual mean squared error (RMSE) for target/source/transported source
            
    """
    # fit the model
    target_model = model_func()
    target_model.fit(target_reps, target_labels)

    # calculate the stats
    target_pred_labels = target_model.predict(target_reps)
    target_mae = metrics.mean_absolute_error(target_labels, target_pred_labels)
    target_mse = metrics.mean_squared_error(target_labels, target_pred_labels)
    target_rmse = np.sqrt(metrics.mean_squared_error(target_labels, target_pred_labels))

    source_pred_labels = target_model.predict(source_reps)
    source_mae = metrics.mean_absolute_error(source_labels, source_pred_labels)
    source_mse = metrics.mean_squared_error(source_labels, source_pred_labels)
    source_rmse = np.sqrt(metrics.mean_squared_error(source_labels, source_pred_labels))

    trans_source_pred_labels = target_model.predict(trans_source_reps)
    trans_source_mae = metrics.mean_absolute_error(source_labels, trans_source_pred_labels)
    trans_source_mse = metrics.mean_squared_error(source_labels, trans_source_pred_labels)
    trans_source_rmse =  np.sqrt(metrics.mean_squared_error(source_labels, trans_source_pred_labels))

    return target_mae, target_mse, target_rmse, source_mae, source_mse, source_rmse,\
        trans_source_mae, trans_source_mse, trans_source_rmse


def cal_stats_emb_ordered(target_clf, aug_target_clf, test_reps, test_labels):
    """ 
    Calculates the accuracy statistics for embedding-based simulation with ordered reponse

    :param function target_clf: the target function trained by target data
    :param function aug_target_clf: the target function target by target data and (augmented) transported source data
    :param list[float] test_reps: the 1D test data representations
    :param list[float] test_labels: test labels

    returns
        - MAE, MSE, RMSE of target model/augmented target model
    """
    target_pred_labels = target_clf.predict(test_reps)
    target_mae = metrics.mean_absolute_error(test_labels, target_pred_labels)
    target_mse = metrics.mean_squared_error(test_labels, target_pred_labels)
    target_rmse = np.sqrt(metrics.mean_squared_error(test_labels, target_pred_labels))
    
    aug_target_pred_labels = aug_target_clf.predict(test_reps)
    aug_target_mae = metrics.mean_absolute_error(test_labels, aug_target_pred_labels)
    aug_target_mse = metrics.mean_squared_error(test_labels, aug_target_pred_labels)
    aug_target_rmse = np.sqrt(metrics.mean_squared_error(test_labels, aug_target_pred_labels))

    return target_mae, target_mse, target_rmse, aug_target_mae, aug_target_mse, aug_target_rmse


def train_model(reps, labels, model_func):
    """ 
    Trains a model using reps and labels and returns the model
    """
    clf = model_func()
    clf.fit(reps, labels)
    return clf


# def compute_transfer_score(trans_source_reps, source_labels, model_func, target_model):
#     """ 
#     deprecated 
#     Computes the transfer score

#     :param function model_func: the function to model the relationship between transported source reps and source labels
#     :param function target_model: the model trained by target representations and target labels
#     """
#     trans_source_model = train_model(trans_source_reps, source_labels, model_func)
#     transfer_score = mutual_info_score(target_model.coef_, trans_source_model.coef_)
#     return transfer_score



""" 
Wrap up everything for binary labels
"""

def entire_proc_binary(sim_func, custom_train_reps, model_func, max_iter, transfer_score=False):
    """ 
    Executes the entire procedure including
        - generate target sequences, target labels, source sequences and source labels
        - generate target representations and source representations
        - transport source representations to target representations
        - train logistic regression model using target representations and target expires
        - calculate the transferability score by computing the KL divergence between the two model weights
        - calculate accuracy statistics for targets, sources and transported sources 

    :param function sim_func: simulation function
    :param function custom_train_reps: customized deep patient function for training representations
    :param function model_func: the function to model the relationship bewteen representations and response
    :param int max_iter: maximum number of iteration for Sinkhorn transport
    :param bool transfer_score: wheter to compute transferability score, default False
    :returns: the accuracy scores, and the transferability score
    """
    target_seqs, target_labels, source_seqs, source_labels = sim_func()
    target_reps, source_reps = custom_train_reps(target_seqs, source_seqs)
    trans_source_reps = trans_source2target(source_reps, target_reps, max_iter=max_iter)

    target_model = train_model(target_reps, target_labels, model_func)

    target_accuracy, target_precision, target_recall, target_f1, \
        source_accuracy, source_precision, source_recall, source_f1, \
        trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1 = \
        cal_stats_binary(target_reps, target_labels, source_reps, source_labels, trans_source_reps, target_model)

    if transfer_score:
        transfer_score = compute_transfer_score(trans_source_reps, source_labels, model_func)
        return target_accuracy, target_precision, target_recall, target_f1, \
            source_accuracy, source_precision, source_recall, source_f1, \
            trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1, \
            transfer_score
    
    return target_accuracy, target_precision, target_recall, target_f1, \
        source_accuracy, source_precision, source_recall, source_f1, \
        trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1
    

""" 
Wrap up everything for continuous labels
"""

def entire_proc_cts(sim_func, custom_train_reps, model_func, max_iter):
    """ 
    Executes the entire procedure including
        - generate target sequences, target labels, source sequences and source labels
        - generate target representations and source representations
        - transport source representations to target representations
        - train regression model using target representations and target expires
        - calculate accuracy statistics for targets, sources and transported sources

    :param function sim_func: simulation function
    :param function custom_train_reps: customized deep patient function for training representations
    :param function model_func: the function to model the relationship bewteen representations and response
    :param int max_iter: maximum number of iterations for Sinkhorn OT
    :returns: the accuracy scores
    """
    target_seqs, target_labels, source_seqs, source_labels = sim_func()
    target_reps, source_reps = custom_train_reps(target_seqs, source_seqs)
    trans_source_reps = trans_source2target(source_reps, target_reps, max_iter=max_iter)
    
    target_mae, target_mse, target_rmse, source_mae, source_mse, source_rmse, \
        trans_source_mae, trans_source_mse, trans_source_rmse = \
        cal_stats_cts(target_reps, target_labels, source_reps, source_labels, trans_source_reps, model_func)
    return target_mae, target_mse, target_rmse,  source_mae, source_mse, source_rmse, \
        trans_source_mae, trans_source_mse, trans_source_rmse


def train_model(reps, labels, model_func = linear_model.LinearRegression): 
    """ 
    Train a model using a model function and by representations reps and labels

    :param function model_func: the model function, e.g. linear_model.LinearRegression

    :returns:
        - the learned linear model
    """
    clf = model_func()
    clf.fit(reps, labels)
    return clf


def entire_proc_emb_ordered(sim_func, custom_train_reps, max_iter):
    """ 
    Run the entire procedure for embedding simulated and ordered response
    """


    target_features, target_labels, test_features, test_labels, source_features, source_labels = sim_func()
    target_reps, source_reps, test_reps = custom_train_reps(target_features, source_features, test_features)
    trans_source_reps = trans_source2target(source_reps, target_reps, max_iter=max_iter)

    target_clf = train_model(target_reps, target_labels)
    aug_target_clf = \
        train_model(np.append(target_reps, trans_source_reps, axis = 0), np.append(target_labels, source_labels, axis=0))

    target_mae, target_mse, target_rmse, aug_target_mae, aug_target_mse, aug_target_rmse = \
        cal_stats_emb_ordered(target_clf, aug_target_clf, test_reps, test_labels)

    return target_mae, target_mse, target_rmse, aug_target_mae, aug_target_mse, aug_target_rmse


""" 
Run entire procedure on multiple simulations and print accuracy statistics, \
    for binary labels
"""

def run_proc_multi(sim_func, custom_train_reps, model_func, max_iter = None, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for binary labels

    :param function model_func: the function to model the relationship between representations and responses
    :param bool filter: whether to filter out source accuracies > 0.7
    :param int max_iter: maximum number of iterations for Sinkhorn transport

    :returns: vectors of accuracy statistics of multiple rounds
    """
    
    target_accuracies = []
    target_precisions = [] 
    target_recalls = [] 
    target_f1s = []
    source_accuracies = []
    source_precisions = []
    source_recalls = [] 
    source_f1s = []
    trans_source_accuracies = []
    trans_source_precisions = []
    trans_source_recalls = []
    trans_source_f1s = []

    for i in range(n_times):
        print(f"iteration: {i}")
        # init accuracies
        target_accuracy = None
        target_precision = None
        target_recall = None
        target_f1 = None
        source_accuracy = None
        source_precision = None
        source_recall = None
        source_f1 = None
        trans_source_accuracy = None
        trans_source_precision = None
        trans_source_recall = None
        trans_source_f1 = None

        try:
            target_accuracy, target_precision, target_recall, target_f1, \
            source_accuracy, source_precision, source_recall, source_f1, \
            trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1 = \
                    entire_proc_binary(sim_func, custom_train_reps, model_func, max_iter=max_iter)

        except Exception: # most likely only one label is generated for the examples
            print("exception 1")
            continue

        # if domain 2 data performs better using the model trained by domain 1 data, \
        # there is no need to transport
        if target_accuracy <= source_accuracy: 
            print("exception 2")
            continue

        # denominator cannot be 0
        min_deno = 0.001
        target_accuracy = max(target_accuracy, min_deno)
        target_precision = max(target_precision, min_deno)
        target_recall = max(target_recall, min_deno)
        target_f1 = max(target_f1, min_deno)
        source_accuracy = max(source_accuracy, min_deno)
        source_precision = max(source_precision, min_deno)
        source_recall = max(source_recall, min_deno)
        source_f1 = max(source_f1, min_deno)
        trans_source_accuracy = max(trans_source_accuracy, min_deno)
        trans_source_precision = max(trans_source_precision, min_deno)
        trans_source_recall = max(trans_source_recall, min_deno)
        trans_source_f1 = max(trans_source_f1, min_deno)

        target_accuracies.append(target_accuracy)
        target_precisions.append(target_precision)
        target_recalls.append(target_recall)
        target_f1s.append(target_f1)
        source_accuracies.append(source_accuracy)
        source_precisions.append(source_precision)
        source_recalls.append(source_recall)
        source_f1s.append(source_f1)
        trans_source_accuracies.append(trans_source_accuracy)
        trans_source_precisions.append(trans_source_precision)
        trans_source_recalls.append(trans_source_recall) 
        trans_source_f1s.append(trans_source_f1)
    return target_accuracies, target_precisions, target_recalls, target_f1s, \
        source_accuracies, source_precisions, source_recalls, source_f1s, \
        trans_source_accuracies, trans_source_precisions, trans_source_recalls, trans_source_f1s


""" 
Run entire procedure on multiple simulations and print accuracy statistics, \
    for continuous labels
"""

def run_proc_multi_cts(sim_func, custom_train_reps, model_func, max_iter = None, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for continuous labels

    :param function model_func: the function to model the relationship between representations and responses
    :param int max_iter: maximum number of iterations for Sinkhorn OT

    :returns: vectors of accuracy statistics of multiple rounds
    """
    
    target_maes = []
    target_mses = [] 
    target_rmses = [] 
    source_maes = []
    source_mses = []
    source_rmses = [] 
    trans_source_maes = []
    trans_source_mses = []
    trans_source_rmses = []

    for _ in range(n_times):
        # init accuracies
        target_mae = None
        target_mse = None
        target_rmse = None 
        source_mae = None
        source_mse = None
        source_rmse = None 
        trans_source_mae = None
        trans_source_mse = None
        trans_source_rmse = None


        try:
            target_mae, target_mse, target_rmse, source_mae, source_mse, source_rmse, \
                trans_source_mae, trans_source_mse, trans_source_rmse = \
                    entire_proc_cts(sim_func, custom_train_reps, model_func, max_iter)
                    
        except Exception: # most likely only one label is generated for the examples
            print("exception 1")
            continue

        # if domain 2 data performs better using the model trained by domain 1 data, \
        # there is no need to transport
        if target_mae >= source_mae: 
            print("exception 2")
            continue

        target_maes.append(target_mae)
        target_mses.append(target_mse)
        target_rmses.append(target_rmse)
        source_maes.append(source_mae)
        source_mses.append(source_mse)
        source_rmses.append(source_rmse)
        trans_source_maes.append(trans_source_mae)
        trans_source_mses.append(trans_source_mse)
        trans_source_rmses.append(trans_source_rmse)
    return target_maes, target_mses, target_rmses,  source_maes, source_mses, source_rmses, \
        trans_source_maes, trans_source_mses, trans_source_rmses


def run_proc_multi_emb_ordered(sim_func, custom_train_reps, max_iter, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for embedding-based ordered response

    :returns: vectors of accuracy statistics of multiple rounds
    """

    target_maes = []
    target_mses = [] 
    target_rmses = [] 
    aug_target_maes = []
    aug_target_mses = [] 
    aug_target_rmses = [] 

    for _ in range(n_times):
        # init accuracies
        target_mae = None
        target_mse = None
        target_rmse = None 
        aug_target_mae = None
        aug_target_mse = None
        aug_target_rmse = None 

        target_mae, target_mse, target_rmse, aug_target_mae, aug_target_mse, aug_target_rmse = \
            entire_proc_emb_ordered(sim_func, custom_train_reps, max_iter)

        target_maes.append(target_mae)
        target_mses.append(target_mse)
        target_rmses.append(target_rmse)
        aug_target_maes.append(aug_target_mae)
        aug_target_mses.append(aug_target_mse)
        aug_target_rmses.append(aug_target_rmse)
     
    return target_maes, target_mses, target_rmses,  aug_target_maes, aug_target_mses, aug_target_rmses


""" 
Constructs a dataframe to demonstrate the accuracy statistics for binary labels
"""

def save_scores(target_accuracies, target_precisions, target_recalls, target_f1s, \
        source_accuracies, source_precisions, source_recalls, source_f1s, \
        trans_source_accuracies, trans_source_precisions, trans_source_recalls, trans_source_f1s, file_path):
    """ 
    Save accuracy statistics to file path
    """
    # construct dataframe
    score_df = pd.DataFrame()
    score_df['target_accuracy'] = target_accuracies
    score_df['target_precision'] = target_precisions
    score_df['target_recall'] = target_recalls
    score_df['target_f1'] = target_f1s
    score_df['source_accuracy'] = source_accuracies
    score_df['source_precision'] = source_precisions
    score_df['source_recall'] = source_recalls
    score_df['source_f1'] = source_f1s
    score_df['trans_source_accuracy'] = trans_source_accuracies
    score_df['trans_source_precision'] = trans_source_precisions
    score_df['trans_source_recall'] = trans_source_recalls
    score_df['trans_source_f1'] = trans_source_f1s
    # save
    score_df.to_csv(file_path, index=None, header=True)



""" 
Constructs a dataframe to demonstrate the accuracy statistics for continuous labels
"""

def save_scores_cts(target_maes, target_mses, target_rmses,  source_maes, source_mses, source_rmses, \
        trans_source_maes, trans_source_mses, trans_source_rmses, file_path):
    """ 
    Save accuracy statistics to file path
    """
    # construct dataframe
    score_df = pd.DataFrame()
    score_df['target_mae'] = target_maes
    score_df['target_mse'] = target_mses
    score_df['target_rmse'] = target_rmses
    score_df['source_mae'] = source_maes
    score_df['source_mse'] = source_mses
    score_df['source_rmse'] = source_rmses
    score_df['trans_source_mae'] = trans_source_maes
    score_df['trans_source_mse'] = trans_source_mses
    score_df['trans_source_rmse'] = trans_source_rmses

    # save
    score_df.to_csv(file_path, index=None, header=True)


""" 
Constructs a dataframe to demonstrate the accuracy statistics for embedding-based ordered response
"""

def save_scores_emb_ordered(target_maes, target_mses, target_rmses,  aug_target_maes, aug_target_mses, aug_target_rmses, file_path):
    """ 
    Save accuracy statistics to file path
    """
    # construct dataframe
    score_df = pd.DataFrame()
    score_df['target_mae'] = target_maes
    score_df['target_mse'] = target_mses
    score_df['target_rmse'] = target_rmses
    score_df['aug_target_mae'] = aug_target_maes
    score_df['aug_target_mse'] = aug_target_mses
    score_df['aug_target_rmse'] = aug_target_rmses

    # save
    score_df.to_csv(file_path, index=None, header=True)



""" 
Box plot of simulation result statistics
"""

def box_plot_binary_short(score_path, save_path = None):
    """ 
    Box plot of the scores in score dataframe stored in score_path for binary labels. \
        Specifically, we plot the box plots of 
        - precision/recall of source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of source

    :param str score_path: the path to scores.csv
    :param str save_path: the path to save plot
    """

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    target_precision = scores_df['target_precision']
    target_recall = scores_df['target_recall']
    target_f1 = scores_df['target_f1']

    source_precision = scores_df['source_precision']
    source_recall = scores_df['source_recall']
    source_f1 = scores_df['source_f1']

    trans_source_precision = scores_df['trans_source_precision']
    trans_source_recall = scores_df['trans_source_recall']
    trans_source_f1 = scores_df['trans_source_f1']

    # source to target precision
    source_target_precision = [i / j for i, j in zip(source_precision, target_precision)]

    # transported source to target precision
    trans_source_target_precision = [i / j for i, j in zip(trans_source_precision, target_precision)]

    # transported source to source precision
    trans_source_source_precision = [i / j for i, j in zip(trans_source_precision, source_precision)]
    print("average trans source to source precision is:", np.mean(trans_source_source_precision))
    print("median trans source to source precision is:", np.median(trans_source_source_precision))


    # source to target recall
    source_target_recall = [i / j for i, j in zip(source_recall, target_recall)]

    # transported source to target recall
    trans_source_target_recall = [i / j for i, j in zip(trans_source_recall, target_recall)]

    # transported source to source recall
    trans_source_source_recall = [i / j for i, j in zip(trans_source_recall, source_recall)]
    print("average trans source to source recall is:", np.mean(trans_source_source_recall))
    print("median trans source to source recall is:", np.median(trans_source_source_recall))

    # transported source to source f1
    trans_source_source_f1 = [i / j for i, j in zip(trans_source_f1, source_f1)]
    print("average trans source to source f1 is:", np.mean(trans_source_source_f1))
    print("median trans source to source f1 is:", np.median(trans_source_source_f1))


    



    # Set the figure size
    plt.figure()
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Pandas dataframe
    data = pd.DataFrame({
        'precision': trans_source_source_precision,
        'recall': trans_source_source_recall,
        'f1': trans_source_source_f1
    })

    # Plot the dataframe
    ax = data[['precision', 'recall', 'f1']].plot(kind='box')

    # Plot the baseline
    plt.axhline(y = 1, color = 'r', linestyle = '-')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)


    plt.show()





""" 
Shorter version of box plot of simulation result statistics
"""

def box_plot_label_binary_short(score_path, label_code):
    """ 
    Box plot of the scores in score dataframe stored in score_path for binary labels with respect to the label_code. \
        Specifically, we plot the box plots of 
        - precision/recall of transported source over precision/recall of source

    :param str score_path: the path to scores.csv
    :param str label_code: the ICD code as the response

    Returns:
        - the median of trans source to source precision
        - the median of trans source to source recall
        - the median of the transferability score
    """

    def special_div(x, y):
        """ 
        Special division operation
        """
        if y == 0:
            y = 1e-5
        return x/y

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    source_precision = scores_df['source_precision']
    source_recall = scores_df['source_recall']

    trans_source_precision = scores_df['trans_source_precision']
    trans_source_recall = scores_df['trans_source_recall']

    transfer_score = scores_df['transfer_score']

    # original_score = scores_df['original_score']

    # transported source to source precision
    trans_source2source_precision = [special_div(i, j) for i, j in zip(trans_source_precision, source_precision)]

    # transported source to source recall
    trans_source2source_recall = [special_div(i, j) for i, j in zip(trans_source_recall, source_recall)]

    # # transfer score to original score
    # transfer2original_score = [special_div(i, j) for i, j in zip(transfer_score, original_score)]

    # Set the figure size
    plt.figure()
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.rcParams["figure.autolayout"] = True

    # Pandas dataframe
    data = pd.DataFrame({
        'precision': trans_source2source_precision,
        'recall': trans_source2source_recall,
    })

    # Plot the dataframe
    ax = data[['precision', 'recall']].plot(kind='box', title=f'transported target to target for {label_code}')

    # Plot the baseline
    plt.axhline(y = 1, color = 'r', linestyle = '-')

    # Display the plot
    plt.show()
    return median(trans_source2source_precision), median(trans_source2source_recall), median(transfer_score)


""" 
Shorter version of box plot of simulation result statistics, for continuous response
"""

def box_plot_cts_short(score_path, save_path=None):
    """ 
    Box plot of the scores in score dataframe stored in score_path for binary labels. \
        Specifically, we plot the box plots of 
        - mae/rmse of transported source over accuracy/precision/recall of source

    :param str score_path: the path to scores.csv
    :param str response_name: the name of response

    Returns:
        - the medians of trans source to source mae
        - the medians of trans source to source rmse
    """

    def special_div(x, y):
        """ 
        Special division operation
        """
        if y == 0:
            y = 1e-5
        return x/y

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    source_mae = scores_df['source_mae']
    source_rmse = scores_df['source_rmse']

    trans_source_mae = scores_df['trans_source_mae']
    trans_source_rmse = scores_df['trans_source_rmse']

    # transported source to source mae
    trans_source_source_mae = [special_div(i, j) for i, j in zip(trans_source_mae, source_mae)]

    # transported source to source rmse
    trans_source_source_rmse = [special_div(i, j) for i, j in zip(trans_source_rmse, source_rmse)]

    # Set the figure size
    plt.figure()
    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    # plt.rcParams["figure.autolayout"] = True

    # Pandas dataframe
    data = pd.DataFrame({
        'MAE': trans_source_source_mae,
        'RMSE': trans_source_source_rmse
    })

    # Plot the dataframe
    ax = data[['MAE', 'RMSE']].plot(kind='box')

    # Plot the baseline
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')


    plt.show()
    
    return median(trans_source_source_mae), median(trans_source_source_rmse)


""" 
Histogram plot of simulation result statistics for continuous labels
"""
def hist_plot_cts(score_path):
    """ 
    histogram plot of the scores in score dataframe stored in score_path for binary labels. \
        Specifically, we plot the box plots of 
        - mae/mse/rmse of source over mae/mse/rmse of target
        - mae/mse/rmse of transported source over mae/mse/rmse of target
        - mae/mse/rmse of transported source over mae/mse/rmse of source

    :param str score_path: the path to scores.csv
    """

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    target_mae = scores_df['target_mae']
    target_rmse = scores_df['target_rmse']

    source_mae = scores_df['source_mae']
    source_rmse = scores_df['source_rmse']

    trans_source_mae = scores_df['trans_source_mae']
    trans_source_rmse = scores_df['trans_source_rmse']

    fig = plt.figure(figsize=(16,10))
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'fuchsia'}



    # transported source to source mae
    trans_source_source_mae = [i / j for i, j in zip(trans_source_mae, source_mae)]

    # transported source to source rmse
    trans_source_source_rmse = [i / j for i, j in zip(trans_source_rmse, source_rmse)]


    bin_width = 0.01
    plt.subplot(1, 2, 1)
    plt.hist(trans_source_source_mae, \
        bins=np.arange(min(trans_source_source_mae), max(trans_source_source_mae) + bin_width, bin_width))
    plt.title("trans source to source accuracy ratio histogram")

    
    plt.subplot(1, 2, 2)
    plt.hist(trans_source_source_rmse , \
        bins=np.arange(min(trans_source_source_rmse), max(trans_source_source_rmse) + bin_width, bin_width))
    plt.title("trans source to source rmse ratio histogram")

    print("average trans source to source mae is {:.1%}".format(np.mean(trans_source_source_mae)))
    print("median trans source to source mae is {:.1%}".format(np.median(trans_source_source_mae)))
    print("average trans source to source rmse is {:.1%}".format(np.mean(trans_source_source_rmse)))
    print("median trans source to source rmse f1 is {:.1%}".format(np.median(trans_source_source_rmse)))

    plt.tight_layout()
    plt.show()


""" 
Histogram plot of simulation result statistics
"""
def hist_plot(score_path, filter = True):
    """ 
    histogram plot of the scores in score dataframe stored in score_path for binary labels. \
        Specifically, we plot the box plots of 
        - precision/recall of source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of source

    :param str score_path: the path to scores.csv
    :param bool filter: filter out scores where source accuracy is greater than > 0.7 (small room for improvement)
    """

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    target_accuracy = scores_df['target_accuracy']
    target_f1 = scores_df['target_f1']

    source_accuracy = scores_df['source_accuracy']
    source_f1 = scores_df['source_f1']

    trans_source_accuracy = scores_df['trans_source_accuracy']
    trans_source_f1 = scores_df['trans_source_f1']

    if filter:
        delete_indices = []
        high_acc_thres = 0.7
        for i in range(len(source_accuracy)):
            if source_accuracy[i] > high_acc_thres:
                delete_indices.append(i)
        target_accuracy = np.delete(list(target_accuracy), delete_indices)
        target_f1 = np.delete(list(target_f1), delete_indices)
        source_accuracy = np.delete(list(source_accuracy), delete_indices)
        source_f1 = np.delete(list(source_f1), delete_indices)
        trans_source_accuracy = np.delete(list(trans_source_accuracy), delete_indices)
        trans_source_f1 = np.delete(list(trans_source_f1), delete_indices)
    

    trans_source_source_accuracy_incre =  [i - j for i, j in zip(trans_source_accuracy, source_accuracy)]
    trans_source_source_f1_incre =  [i - j for i, j in zip(trans_source_f1, source_f1)]

    print("average trans target to target accuracy increment is {:.1%}".format(np.mean(trans_source_source_accuracy_incre)))
    print("median trans target to target accuracy increment is {:.1%}".format(np.median(trans_source_source_accuracy_incre)))
    print("average trans target to target accuracy f1 is {:.1%}".format(np.mean(trans_source_source_f1_incre)))
    print("median trans target to target accuracy f1 is {:.1%}".format(np.median(trans_source_source_f1_incre)))

    fig = plt.figure(figsize=(16,16))
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'fuchsia'}

    # source to target accuracy
    source_target_accuracy = [i / j for i, j in zip(source_accuracy, target_accuracy)]

    # transported source to target accuracy
    trans_source_target_accuracy = [i / j for i, j in zip(trans_source_accuracy, target_accuracy)]

    # transported source to source accuracy
    trans_source_source_accuracy = [i / j for i, j in zip(trans_source_accuracy, source_accuracy)]


    # source to target accuracy
    source_target_f1 = [i / j for i, j in zip(source_f1, target_f1)]

    # transported source to target accuracy
    trans_source_target_f1 = [i / j for i, j in zip(trans_source_f1, target_f1)]

    # transported source to source accuracy
    trans_source_source_f1 = [i / j for i, j in zip(trans_source_f1, source_f1)]

    bin_width = 0.01
    plt.subplot(3, 3, 1)
    plt.hist(trans_source_source_accuracy_incre, \
        bins=np.arange(min(trans_source_source_accuracy_incre), max(trans_source_source_accuracy_incre) + bin_width, bin_width))
    plt.title("trans target to target accuracy increment histogram")

    
    plt.subplot(3, 3, 2)
    plt.hist(trans_source_source_f1_incre , \
        bins=np.arange(min(trans_source_source_f1_incre), max(trans_source_source_f1_incre) + bin_width, bin_width))
    plt.title("trans target to target f1 increment histogram")

    plt.tight_layout()
    plt.show()


def vis_emb_dim2_ordered(target_reps, target_labels, source_reps, source_labels, trans_source_reps, model):
    """ 
    Visualize the embedding space of dimension 2 of the target data, source data and transported source data \
        for ordered response (continuous and discrete response)
    
    :param function model: the trained model
    """

    fig = plt.figure(figsize=(15, 15))
    ax = Axes3D(fig)
    ax.scatter(source_reps[:, 0], source_reps[:, 1], source_labels, marker='+', color='red', label="Source Samples")
    ax.scatter(target_reps[:, 0], target_reps[:, 1], target_labels, marker='o', color = "green", label='Target samples')
    x_pred = np.linspace(-0.6, 1.4, num=100)
    y_pred = np.linspace(-0.4, 1, num=100)
    z_pred = np.linspace(-0.6, 1.4, num=100)
    xx_pred, yy_pred, zz_pred = np.meshgrid(x_pred, y_pred, z_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten(), zz_pred.flatten()]).T
    predicted = model.predict(model_viz)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')

    ax.scatter(trans_source_reps[:, 0], trans_source_reps[:, 1], source_labels, marker='+', color="blue", label='Transported source samples')
    plt.legend()
    plt.show()


def vis_boundary(reps, labels, model):
    # construct colors
    colors = []
    unique_colors = list(set(labels))
    for label in labels:
        if label == unique_colors[0]:
            colors.append('red')
        else:
            colors.append('blue')
    
    class_1_indices = [i for i,val in enumerate(colors) if val=="red"]
    class_2_indices = [i for i,val in enumerate(colors) if val=="blue"]

    fig = plt.figure(figsize=(15, 15))
    ax = Axes3D(fig)
    ax.scatter(reps[class_1_indices, 0], reps[class_1_indices, 1], reps[class_1_indices, 2], edgecolors="red", alpha = 0.5, s=30, facecolors='none', label="class 1")
    ax.scatter(reps[class_2_indices, 0], reps[class_2_indices, 1], reps[class_2_indices, 2], edgecolors="blue", alpha = 0.5, s=10, facecolors='none', label = "class 2")
    
    x_pred = np.linspace(-0.6, 1.4, num=100)
    y_pred = np.linspace(-0.4, 1, num=100)
    z_pred = np.linspace(-0.6, 1.4, num=100)

    xx_pred, yy_pred, zz_pred = np.meshgrid(x_pred, y_pred, z_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten(), zz_pred.flatten()]).T
    predicted = model.predict(model_viz)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=5, edgecolor='#70b3f0')

    plt.legend()
    plt.show()


def vis_emb_dim2_unordered(target_reps, target_labels, source_reps, source_labels, trans_source_reps, save_path = None):
    """ 
    Visualize the embedding space of dimension 2 of the target data, source data and transported source data \
        for unordered reponse (categorical response)
    
        :param str save_path: path to save the figure
    """
    
    pl.figure(1, figsize=(15, 5))
    pl.subplot(1, 3, 1)
    pl.scatter(source_reps[:, 0], source_reps[:, 1], c=source_labels, alpha = 0.5, marker='o')
    pl.xticks([])
    pl.yticks([])
    # pl.legend(loc=0)
    pl.title('(a) Target embedding', fontweight='bold', loc='left', fontsize=20)

    pl.figure(1, figsize=(15, 5))
    pl.subplot(1, 3, 2)
    pl.scatter(target_reps[:, 0], target_reps[:, 1], c=target_labels, marker='+')
    pl.xticks([])
    pl.yticks([])
    # pl.legend(loc=0)
    pl.title('(b) Source embedding', fontweight='bold', loc='left', fontsize=20)

    pl.figure(1, figsize=(15, 5))
    pl.subplot(1, 3, 3)
    pl.scatter(trans_source_reps[:, 0], trans_source_reps[:, 1],c=target_labels, alpha = 0.5, marker='o')
    pl.xticks([])
    pl.yticks([])
    # pl.legend(loc=0)
    pl.title('(c) Transported target embedding', fontweight='bold', loc='left', fontsize=20)
    pl.tight_layout()

    if save_path is not None:
        pl.savefig(save_path, bbox_inches = 'tight')
    pl.show()


def vis_emb_dim1_ordered(target_reps, target_labels, source_reps, source_labels, \
    trans_source_reps, test_reps, test_labels, target_model, aug_target_model, low=0, high=200):
    """
    Visualize the embedding space of dimension 1 of the target data, source data and transported source data \
        for ordered reponse (e.g. continuous response, discrete response),
        also visualize the learned functions
    :param function target_model: the model trained by target data only
    :param function aug_target_model: the model trained by target data and transported source data
    :param float low: the lower bound for visualizing the two functions, default 0
    :param float high: the upper bound for visualizing the two functions, default 200

    """
    plt.figure(1, figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.scatter(source_reps, source_labels,  alpha = 0.5, marker='o', c='cornflowerblue')
    plt.title('Source embedding')
    plt.xlabel("embedding dim")
    plt.ylabel("response")

    plt.subplot(1, 4, 2)
    target_points = plt.scatter(target_reps, target_labels, alpha=0.5, marker='+', c='red')
    test_points = plt.scatter(test_reps, test_labels, alpha=0.5, marker='v', c='black')
    # visualize functions
    x = np.array([[low], [high]])
    target_y = target_model.predict(x)
    plt.plot(x, target_y, 'gold')
    plt.title('Target embedding')
    plt.xlabel("embedding dim")
    plt.ylabel("response")
    plt.legend((target_points, test_points),
        ('target', 'test'),
        scatterpoints=1,
        loc='lower right',
        ncol=3,
        fontsize=12)

    plt.subplot(1, 4, 3)
    plt.scatter(trans_source_reps, source_labels, alpha = 0.5, marker='o', c='cornflowerblue')
    plt.title('Transported source embedding')
    plt.xlabel("embedding dim")
    plt.ylabel("response")

    plt.subplot(1, 4, 4)
    trans_source_points = plt.scatter(trans_source_reps, source_labels, alpha = 0.5, marker='o', c='cornflowerblue')
    target_points = plt.scatter(target_reps, target_labels, alpha=0.5, marker='+', c='red')
    test_points = plt.scatter(test_reps, test_labels, alpha=0.5, marker='v', c='black')
    aug_target_y = aug_target_model.predict(x)
    plt.plot(x, aug_target_y, 'gold')
    plt.title('Target and transported source embedding')
    plt.legend((trans_source_points, target_points, test_points),
           ('trans source', 'target', 'test'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=9)
    plt.xlabel("embedding dim")
    plt.ylabel("response")
    plt.tight_layout()
    plt.show()


""" 
Histogram plot of simulation result statistics for embedding-based ordered labels
"""
def hist_plot_ordered_emb(score_path):
    """ 
    histogram plot of the scores in score dataframe stored in score_path for ordered labels generated by embeddings. \
        Specifically, we plot the box plots of 
        - mae/mse/rmse of source over mae/mse/rmse of target
        - mae/mse/rmse of source over mae/mse/rmse of augmented target
 

    :param str score_path: the path to scores.csv
    """

    scores_df = pd.read_csv(score_path, index_col=None, header=0)

    target_mae = scores_df['target_mae']
    target_rmse = scores_df['target_rmse']

    aug_target_mae = scores_df['aug_target_mae']
    aug_target_rmse = scores_df['aug_target_rmse']


    fig = plt.figure(figsize=(16,16))
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'fuchsia'}

    # transported source to source mae
    aug_target_target_mae = [i / j for i, j in zip(aug_target_mae, target_mae)]

    # transported source to source rmse
    aug_target_target_rmse = [i / j for i, j in zip(aug_target_rmse, target_rmse)]

    bin_width = 0.01
    plt.subplot(3, 3, 1)
    plt.hist(aug_target_target_mae, \
        bins=np.arange(min(aug_target_target_mae), max(aug_target_target_mae) + bin_width, bin_width))
    plt.title("aug target to target mae ratio histogram")

    
    plt.subplot(3, 3, 2)
    plt.hist(aug_target_target_rmse , \
        bins=np.arange(min(aug_target_target_rmse), max(aug_target_target_rmse) + bin_width, bin_width))
    plt.title("aug target to target rmse ratio histogram")

    print("average trans source to source mae is {:.1%}".format(np.mean(aug_target_target_mae)))
    print("median trans source to source mae is {:.1%}".format(np.median(aug_target_target_mae)))
    print("average trans source to source rmse is {:.1%}".format(np.mean(aug_target_target_rmse)))
    print("median trans source to source rmse f1 is {:.1%}".format(np.median(aug_target_target_rmse)))

    plt.tight_layout()
    plt.show()
