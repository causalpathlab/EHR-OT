""" 
Common functions for synthetic datasets
"""
import sys
sys.path.append("/home/wanxinli/deep_patient")

import matplotlib.pyplot as plt
import os
import ot
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

""" 
Transport female representations to male representations
"""

def trans_female2male(male_reps, female_reps, max_iter = None):
    """ 
    Optimal transport (without entropy regularization) female representations \
        to male representations

    :param int max_iter: maximum number of iteration for OT
    :returns: transported female representations
    """
    ot_emd = ot.da.EMDTransport()
    if max_iter is not None:
        ot_emd = ot.da.EMDTransport(max_iter=max_iter)
    ot_emd.fit(Xs=female_reps, Xt=male_reps)
    trans_female_reps = ot_emd.transform(Xs=female_reps)
    return trans_female_reps


""" 
Caculate result statistics for binary labels
"""

def cal_stats_binary(male_reps, male_labels, female_reps, female_labels, \
    trans_female_reps, max_iter = None):
    """ 
    Calculate accuracy statistics based on logistic regression between the \
        patient representations and label labels
    This function is for binary labels

    :param max_iter: maxmium number of iterations for the logistic model
    
    :returns: using the male model,\
        - accuracy for male/female/transported female
        - precision for male/female/transported female
        - recall for male/female/transported female
            
    """
    # fit the model
    male_logit_model = linear_model.LogisticRegression()
    if max_iter is not None:
        male_logit_model = linear_model.LogisticRegression(max_iter=max_iter)
    male_logit_model.fit(male_reps, male_labels)

    # calculate the stats
    male_pred_labels = male_logit_model.predict(male_reps)
    male_accuracy = accuracy_score(male_labels, male_pred_labels)
    male_precision = precision_score(male_labels, male_pred_labels)
    male_recall = recall_score(male_labels, male_pred_labels)

    female_pred_labels = male_logit_model.predict(female_reps)
    female_accuracy = accuracy_score(female_labels, female_pred_labels)
    female_precision = precision_score(female_labels, female_pred_labels)
    female_recall = recall_score(female_labels, female_pred_labels)

    trans_female_pred_labels = male_logit_model.predict(trans_female_reps)
    trans_female_accuracy = accuracy_score(female_labels, trans_female_pred_labels)
    trans_female_precision = precision_score(female_labels, trans_female_pred_labels)
    trans_female_recall = recall_score(female_labels, trans_female_pred_labels)


    return male_accuracy, male_precision, male_recall, \
        female_accuracy, female_precision, female_recall, \
        trans_female_accuracy, trans_female_precision, trans_female_recall


""" 
Wrap up everything for binary labels
"""

def entire_proc_binary(sim_func, custom_train_reps):
    """ 
    Executes the entire procedure including
        - generate male sequences, male labels, female sequences and female labels
        - generate male representations and female representations
        - transport female representations to male representations
        - train logistic regression model using male representations and male expires
        - calculate accuracy statistics for males, females and transported females

    :param function sim_func: simulation function
    :param function custom_train_reps: customized deep patient function for training representations
    :returns: the accuracy scores
    """
    male_seqs, male_labels, female_seqs, female_labels = sim_func()
    male_reps, female_reps = custom_train_reps(male_seqs, female_seqs)
    trans_female_reps = trans_female2male(male_reps, female_reps)
    male_accuracy, male_precision, male_recall, \
        female_accuracy, female_precision, female_recall, \
        trans_female_accuracy, trans_female_precision, trans_female_recall = \
        cal_stats_binary(male_reps, male_labels, female_reps, female_labels, trans_female_reps)
    return male_accuracy, male_precision, male_recall, \
        female_accuracy, female_precision, female_recall, \
        trans_female_accuracy, trans_female_precision, trans_female_recall 
    


""" 
Run entire procedure on multiple simulations and print accuracy statistics, \
    for binary labels
"""

def run_proc_multi(sim_func, custom_train_reps, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for binary labels

    :returns: vectors of accuracy statistics of multiple rounds
    """

    male_accuracies = []
    male_precisions = [] 
    male_recalls = [] 
    female_accuracies = []
    female_precisions = []
    female_recalls = [] 
    trans_female_accuracies = []
    trans_female_precisions = []
    trans_female_recalls = []

    for _ in range(n_times):
        male_accuracy, male_precision, male_recall, \
        female_accuracy, female_precision, female_recall, \
        trans_female_accuracy, trans_female_precision, trans_female_recall = \
                entire_proc_binary(sim_func, custom_train_reps)
        male_accuracies.append(male_accuracy)
        male_precisions.append(male_precision)
        male_recalls.append(male_recall)
        female_accuracies.append(female_accuracy)
        female_precisions.append(female_precision)
        female_recalls.append(female_recall)
        trans_female_accuracies.append(trans_female_accuracy)
        trans_female_precisions.append(trans_female_precision)
        trans_female_recalls.append(trans_female_recall) 
    return male_accuracies, male_precisions, male_recalls, \
        female_accuracies, female_precisions, female_recalls, \
        trans_female_accuracies, trans_female_precisions, trans_female_recalls



""" 
Constructs a dataframe to demonstrate the accuracy statistics for binary labels
"""

def save_scores(male_accuracies, male_precisions, male_recalls, \
        female_accuracies, female_precisions, female_recalls, \
        trans_female_accuracies, trans_female_precisions, trans_female_recalls, indir):
    """ 
    Save accuracy statistics to scores.csv in indir
    """
    # construct dataframe
    score_df = pd.DataFrame()
    score_df['male_accuracy'] = male_accuracies
    score_df['male_precision'] = male_precisions
    score_df['male_recall'] = male_recalls
    score_df['female_accuracy'] = female_accuracies
    score_df['female_precision'] = female_precisions
    score_df['female_recall'] = female_recalls
    score_df['trans_female_accuracy'] = trans_female_accuracies
    score_df['trans_female_precision'] = trans_female_precisions
    score_df['trans_female_recall'] = trans_female_recalls
    # save
    score_df.to_csv(os.path.join(indir, "scores.csv"), index=None)


""" 
Box plot of simulation result statistics
"""

def box_plot(scores_path):
    """ 
    Box plot of the scores in score dataframe stored in scores_path for binary labels. \
        Specifically, we plot the box plots of 
        - precision/recall of female over accuracy/precision/recall of male
        - precision/recall of transported female over accuracy/precision/recall of male
        - precision/recall of transported female over accuracy/precision/recall of female

    :param scores_path: the path to scores.csv
    """

    scores_df = pd.read_csv(scores_path, index_col=None, header=0)

    male_accuracy = scores_df['male_accuracy']
    male_precision = scores_df['male_precision']
    male_recall = scores_df['male_recall']

    female_accuracy = scores_df['female_accuracy']
    female_precision = scores_df['female_precision']
    female_recall = scores_df['female_recall']

    trans_female_accuracy = scores_df['trans_female_accuracy']
    trans_female_precision = scores_df['trans_female_precision']
    trans_female_recall = scores_df['trans_female_recall']

    fig = plt.figure(figsize=(16,16))
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'fuchsia'}

    # female to male accuracy
    female_male_accuracy = [i / j for i, j in zip(female_accuracy, male_accuracy)]
    plt.subplot(3, 3, 1)
    plt.boxplot(female_male_accuracy, flierprops=flierprops)
    plt.title("female accuracy to \n male accuracy")

    # transported female to male accuracy
    trans_female_male_accuracy = [i / j for i, j in zip(trans_female_accuracy, male_accuracy)]
    plt.subplot(3, 3, 2)
    plt.boxplot(trans_female_male_accuracy, flierprops=flierprops)
    plt.title("transported female \n accuracy to \n male accuracy")

    # transported female to female accuracy
    trans_female_female_accuracy = [i / j for i, j in zip(trans_female_accuracy, female_accuracy)]
    plt.subplot(3, 3, 3)
    plt.boxplot(trans_female_female_accuracy, flierprops=flierprops)
    plt.title("transported female \n accuracy to \n female accuracy")

    # female to male precision
    female_male_precision = [i / j for i, j in zip(female_precision, male_precision)]
    plt.subplot(3, 3, 4)
    plt.boxplot(female_male_precision, flierprops=flierprops)
    plt.title("female precision to \n male precision")

    # transported female to male precision
    trans_female_male_precision = [i / j for i, j in zip(trans_female_precision, male_precision)]
    plt.subplot(3, 3, 5)
    plt.boxplot(trans_female_male_precision, flierprops=flierprops)
    plt.title("transported female \n precision to \n male precision")

    # transported female to female precision
    trans_female_female_precision = [i / j for i, j in zip(trans_female_precision, female_precision)]
    plt.subplot(3, 3, 6)
    plt.boxplot(trans_female_female_precision, flierprops=flierprops)
    plt.title("transported female \n precision to \n female precision")

    # female to male recall
    female_male_recall = [i / j for i, j in zip(female_recall, male_recall)]
    plt.subplot(3, 3, 7)
    plt.boxplot(female_male_recall, flierprops=flierprops)
    plt.title("female recall to \n male recall")

    # transported female to male recall
    trans_female_male_recall = [i / j for i, j in zip(trans_female_recall, male_recall)]
    plt.subplot(3, 3, 8)
    plt.boxplot(trans_female_male_recall, flierprops=flierprops)
    plt.title("transported female \n recall to \n male recall")

    # transported female to female recall
    trans_female_female_recall = [i / j for i, j in zip(trans_female_recall, female_recall)]
    plt.subplot(3, 3, 9)
    plt.boxplot(trans_female_female_recall, flierprops=flierprops)
    plt.title("transported female \n recall to \n female recall")

    plt.tight_layout()
    plt.show()


""" 
Caculate result statistics for continuous labels (e.g. duration in hospital)
"""

def cal_stats_cts(male_reps, male_labels, \
    female_reps, female_labels, trans_female_reps):
    """ 
    Calculate accuracy statistics based on linear regression between the \
        patient representations and labels
    This function is for continuous labels
    
    :returns: using the male model,\
        - the coefficient of determination of the predictions of males
        - ... of females
        - ... of transported females
    
    Note that The best possible score is 1.0 and it can be negative \
        (because the model can be arbitrarily worse). \
        A constant model that always predicts the expected value of y, \
        disregarding the input features, would get a score of 0.0.
            
    """
    # fit the model
    male_model = linear_model.LinearRegression()
    male_model = male_model.fit(male_reps, male_labels)

    # calculate the stats
    male_score = male_model.score(male_reps, male_labels)
    female_score = male_model.score(female_reps, female_labels)
    trans_female_score = male_model.score(trans_female_reps, female_labels)

    return male_score, female_score, trans_female_score

