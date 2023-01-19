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
from sklearn.metrics import f1_score

""" 
Transport source representations to target representations
"""

def trans_source2target(source_reps, target_reps, type="balanced", max_iter = None):
    """ 
    Optimal transport (without entropy regularization) source representations \
        to target representations

    :param str type: balanced or unbalanced
    :returns: transported source representations
    """
    trans_source_reps = None
    if type == "balanced":
        ot_emd = ot.da.SinkhornTransport(reg_e=1e-1)
        if max_iter is not None:
            ot_emd = ot.da.SinkhornTransport(reg_e=1e-1, max_iter=max_iter)
        ot_emd.fit(Xs=source_reps, Xt=target_reps)
        trans_source_reps = ot_emd.transform(Xs=source_reps)

    elif type == "unbalanced":
        reg = 0.005
        reg_m_kl = 0.5
        n = source_reps.shape[0]

        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

        M = ot.dist(source_reps, target_reps)
        M /= M.max()

        coupling = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m_kl)
        trans_source_reps = np.matmul(coupling, source_reps)

    return trans_source_reps


""" 
Caculate result statistics for binary labels
"""

def cal_stats_binary(target_reps, target_labels, source_reps, source_labels, \
    trans_source_reps, model_func):
    """ 
    Calculate accuracy statistics based on logistic regression between the \
        patient representations and label labels
    This function is for binary labels

    :param function model_func: the function to model the relationship between \
        representations and reponse
    
    :returns: using the target model,\
        - accuracy for target/source/transported source
        - precision for target/source/transported source
        - recall for target/source/transported source
        - f1 for target/source/transported source
            
    """
    # fit the model
    target_model = model_func()
    target_model.fit(target_reps, target_labels)

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



""" 
Wrap up everything for binary labels
"""

def entire_proc_binary(sim_func, custom_train_reps, model_func, max_iter):
    """ 
    Executes the entire procedure including
        - generate target sequences, target labels, source sequences and source labels
        - generate target representations and source representations
        - transport source representations to target representations
        - train logistic regression model using target representations and target expires
        - calculate accuracy statistics for targets, sources and transported sources

    :param function sim_func: simulation function
    :param function custom_train_reps: customized deep patient function for training representations
    :param function model_func: the function to model the relationship bewteen representations and response
    :param int max_iter: maximum number of iteration for Sinkhorn transport
    :returns: the accuracy scores
    """
    target_seqs, target_labels, source_seqs, source_labels = sim_func()
    target_reps, source_reps = custom_train_reps(target_seqs, source_seqs)
    trans_source_reps = trans_source2target(source_reps, target_reps, max_iter=max_iter)
    
    target_accuracy, target_precision, target_recall, target_f1, \
        source_accuracy, source_precision, source_recall, source_f1, \
        trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1 = \
        cal_stats_binary(target_reps, target_labels, source_reps, source_labels, trans_source_reps, model_func)
    return target_accuracy, target_precision, target_recall, target_f1, \
        source_accuracy, source_precision, source_recall, source_f1, \
        trans_source_accuracy, trans_source_precision, trans_source_recall, trans_source_f1
    

""" 
Wrap up everything for continuous labels
"""

def entire_proc_cts(sim_func, custom_train_reps, model_func):
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
    :returns: the accuracy scores
    """
    target_seqs, target_labels, source_seqs, source_labels = sim_func(num_patient = 50)
    target_reps, source_reps = custom_train_reps(target_seqs, source_seqs)
    trans_source_reps = trans_source2target(source_reps, target_reps)
    
    target_mae, target_mse, target_rmse, source_mae, source_mse, source_rmse, \
        trans_source_mae, trans_source_mse, trans_source_rmse = \
        cal_stats_cts(target_reps, target_labels, source_reps, source_labels, trans_source_reps, model_func)
    return target_mae, target_mse, target_rmse,  source_mae, source_mse, source_rmse, \
        trans_source_mae, trans_source_mse, trans_source_rmse


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

    for _ in range(n_times):
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

def run_proc_multi_cts(sim_func, custom_train_reps, model_func, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for continuous labels

    :param function model_func: the function to model the relationship between representations and responses

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
                    entire_proc_cts(sim_func, custom_train_reps, model_func)
                    
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
Box plot of simulation result statistics
"""

def box_plot(scores_path, filter = True):
    """ 
    Box plot of the scores in score dataframe stored in scores_path for binary labels. \
        Specifically, we plot the box plots of 
        - precision/recall of source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of source

    :param str scores_path: the path to scores.csv
    :param bool filter: filter out scores where source accuracy is greater than > 0.7 (small room for improvement)
    """

    scores_df = pd.read_csv(scores_path, index_col=None, header=0)

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
    print("number of stats is:", len(trans_source_source_accuracy_incre))
    print("number of 0 in incre is:", trans_source_source_accuracy_incre.count(0))
    print("number of elements > 0 is:", np.sum(np.array(trans_source_source_accuracy_incre) > 0, axis=0))
    print("number of elements < 0 is:", np.sum(np.array(trans_source_source_accuracy_incre) < 0, axis=0))
    print("average trans source to source accuracy increment is:", np.mean(trans_source_source_accuracy_incre))
    print("median trans source to source accuracy increment is:", np.median(trans_source_source_accuracy_incre))
    print("average trans source to source accuracy f1 is:", np.mean(trans_source_source_f1_incre ))
    print("median trans source to source accuracy f1 is:", np.median(trans_source_source_f1_incre))

    fig = plt.figure(figsize=(16,16))
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'fuchsia'}

    # source to target accuracy
    source_target_accuracy = [i / j for i, j in zip(source_accuracy, target_accuracy)]

    # transported source to target accuracy
    trans_source_target_accuracy = [i / j for i, j in zip(trans_source_accuracy, target_accuracy)]

    # transported source to source accuracy
    trans_source_source_accuracy = [i / j for i, j in zip(trans_source_accuracy, source_accuracy)]
    print("average trans source to source accuracy is:", np.mean(trans_source_source_accuracy))
    print("median trans source to source accuracy is:", np.median(trans_source_source_accuracy))


    # source to target accuracy
    source_target_f1 = [i / j for i, j in zip(source_f1, target_f1)]

    # transported source to target accuracy
    trans_source_target_f1 = [i / j for i, j in zip(trans_source_f1, target_f1)]

    # transported source to source accuracy
    trans_source_source_f1 = [i / j for i, j in zip(trans_source_f1, source_f1)]
    print("average trans source to source f1 is:", np.mean(trans_source_source_f1))
    print("median trans source to source f1 is:", np.median(trans_source_source_f1))


    plt.subplot(3, 3, 1)
    plt.boxplot(source_target_accuracy, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.title("source accuracy to \n target accuracy")

    
    plt.subplot(3, 3, 2)
    plt.boxplot(trans_source_target_accuracy, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.title("transported source \n accuracy to \n target accuracy")

    
    plt.subplot(3, 3, 3)
    plt.boxplot(trans_source_source_accuracy, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.axhline(y = 1, color = 'b', linestyle = '-')
    plt.title("transported source \n accuracy to \n source accuracy")

    plt.subplot(3, 3, 4)
    plt.boxplot(source_target_f1, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.title("source precision to \n target f1")

    
    plt.subplot(3, 3, 5)
    plt.boxplot(trans_source_target_f1, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.title("transported source \n precision to \n target f1")

    
    plt.subplot(3, 3, 6)
    plt.boxplot(trans_source_source_f1, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.axhline(y = 1, color = 'b', linestyle = '-')
    plt.title("transported source \n precision to \n source f1")

    
    plt.tight_layout()
    plt.show()



""" 
Histogram plot of simulation result statistics for continuous labels
"""
def hist_plot_cts(scores_path):
    """ 
    histogram plot of the scores in score dataframe stored in scores_path for binary labels. \
        Specifically, we plot the box plots of 
        - mae/mse/rmse of source over mae/mse/rmse of target
        - mae/mse/rmse of transported source over mae/mse/rmse of target
        - mae/mse/rmse of transported source over mae/mse/rmse of source

    :param str scores_path: the path to scores.csv
    """

    scores_df = pd.read_csv(scores_path, index_col=None, header=0)

    target_mae = scores_df['target_mae']
    target_rmse = scores_df['target_rmse']

    source_mae = scores_df['source_mae']
    source_rmse = scores_df['source_rmse']

    trans_source_mae = scores_df['trans_source_mae']
    trans_source_rmse = scores_df['trans_source_rmse']

    fig = plt.figure(figsize=(16,16))
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'fuchsia'}



    # transported source to source mae
    trans_source_source_mae = [i / j for i, j in zip(trans_source_mae, source_mae)]

    # transported source to source rmse
    trans_source_source_rmse = [i / j for i, j in zip(trans_source_rmse, source_rmse)]


    bin_width = 0.01
    plt.subplot(3, 3, 1)
    plt.hist(trans_source_source_mae, \
        bins=np.arange(min(trans_source_source_mae), max(trans_source_source_mae) + bin_width, bin_width))
    plt.title("trans source to source accuracy ratio histogram")

    
    plt.subplot(3, 3, 2)
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
def hist_plot(scores_path, filter = True):
    """ 
    histogram plot of the scores in score dataframe stored in scores_path for binary labels. \
        Specifically, we plot the box plots of 
        - precision/recall of source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of target
        - precision/recall of transported source over accuracy/precision/recall of source

    :param str scores_path: the path to scores.csv
    :param bool filter: filter out scores where source accuracy is greater than > 0.7 (small room for improvement)
    """

    scores_df = pd.read_csv(scores_path, index_col=None, header=0)

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

    print("average trans source to source accuracy increment is {:.1%}".format(np.mean(trans_source_source_accuracy_incre)))
    print("median trans source to source accuracy increment is {:.1%}".format(np.median(trans_source_source_accuracy_incre)))
    print("average trans source to source accuracy f1 is {:.1%}".format(np.mean(trans_source_source_f1_incre)))
    print("median trans source to source accuracy f1 is {:.1%}".format(np.median(trans_source_source_f1_incre)))

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
    plt.title("trans source to source accuracy increment histogram")

    
    plt.subplot(3, 3, 2)
    plt.hist(trans_source_source_f1_incre , \
        bins=np.arange(min(trans_source_source_f1_incre), max(trans_source_source_f1_incre) + bin_width, bin_width))
    plt.title("trans source to source f1 increment histogram")

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


def vis_emb_dim2_unordered(target_reps, target_labels, source_reps, source_labels, trans_source_reps):
    """ 
    Visualize the embedding space of dimension 2 of the target data, source data and transported source data \
        for unordered reponse (categorical response)
    """
    pl.figure(1, figsize=(15, 5))
    pl.subplot(1, 3, 1)
    pl.scatter(source_reps[:, 0], source_reps[:, 1], c=source_labels, alpha = 0.5, marker='o')
    pl.xticks([])
    pl.yticks([])
    # pl.legend(loc=0)
    pl.title('Source embedding')

    pl.figure(1, figsize=(15, 5))
    pl.subplot(1, 3, 2)
    pl.scatter(target_reps[:, 0], target_reps[:, 1], c=target_labels, marker='+')
    pl.xticks([])
    pl.yticks([])
    # pl.legend(loc=0)
    pl.title('Target embedding')

    pl.figure(1, figsize=(15, 5))
    pl.subplot(1, 3, 3)
    pl.scatter(trans_source_reps[:, 0], trans_source_reps[:, 1],c=target_labels, alpha = 0.5, marker='o')
    pl.xticks([])
    pl.yticks([])
    # pl.legend(loc=0)
    pl.title('Transported embedding')
    pl.tight_layout()
    pl.show()