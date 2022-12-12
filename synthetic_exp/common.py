""" 
Common functions for synthetic datasets
"""
import sys
sys.path.append("/home/wanxinli/deep_patient")

import csv
from deep_patient.sda import SDA
import numpy as np
import os
import ot
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

""" 
Transport female representations to male representations
"""

def trans_female2male(male_reps, female_reps):
    """ 
    Optimal transport (without entropy regularization) female representations \
        to male representations

    :returns: transported female representations
    """
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=female_reps, Xt=male_reps)
    trans_female_reps = ot_emd.transform(Xs=female_reps)
    return trans_female_reps


""" 
Caculate statistics
"""

def cal_stats(male_reps, male_labels, female_reps, female_labels, trans_female_reps):
    """ 
    Calculate accuracy statistics based on logistic regression between the \
        patient representations and label labels
    
    :returns: accuracy for males using male model, 
            accuracy for females using male model, \
            accuracy for transported females using male model \
            precision for females using male model, \
            recall for females using male model, \
            precision for transported females using male model, \
            recall for transported females using male model, \
            
    """
    # fit the model
    male_logit_model = linear_model.LogisticRegression()
    male_logit_model.fit(male_reps, male_labels)

    # calculate the stats
    male_pred_labels = male_logit_model.predict(male_reps)
    male_precision = precision_score(male_labels, male_pred_labels)
    male_recall = recall_score(male_labels, male_pred_labels)

    female_pred_labels = male_logit_model.predict(female_reps)
    female_precision = precision_score(female_labels, female_pred_labels)
    female_recall = recall_score(female_labels, female_pred_labels)

    trans_female_pred_labels = male_logit_model.predict(trans_female_reps)
    trans_female_precision = precision_score(female_labels, trans_female_pred_labels)
    trans_female_recall = recall_score(female_labels, trans_female_pred_labels)


    return male_precision, male_recall, female_precision, female_recall, trans_female_precision, trans_female_recall


""" 
Wrap up everything
"""

def entire_proc(sim_func, custom_train_reps):
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
    male_precision, male_recall, female_precision, female_recall, trans_female_precision, trans_female_recall  = \
        cal_stats(male_reps, male_labels, female_reps, female_labels, trans_female_reps)
    return male_precision, male_recall, female_precision, female_recall, trans_female_precision, trans_female_recall
    


""" 
Run entire procedure on multiple simulations and print accuracy statistics
"""

def run_proc_multi(sim_func, custom_train_reps, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times)

    :returns: vectors of accuracy statistics of multiple rounds
    """

    male_precisions = [] 
    male_recalls = [] 
    female_precisions = []
    female_recalls = [] 
    trans_female_precisions = []
    trans_female_recalls = []

    for _ in range(n_times):
        male_precision, male_recall, female_precision, female_recall, \
            trans_female_precision, trans_female_recall = \
                entire_proc(sim_func, custom_train_reps)
        male_precisions.append(male_precision)
        male_recalls.append(male_recall)
        female_precisions.append(female_precision)
        female_recalls.append(female_recall)
        trans_female_precisions.append(trans_female_precision)
        trans_female_recalls.append(trans_female_recall) 
    return male_precisions, male_recalls, female_precisions, female_recalls, \
        trans_female_precisions, trans_female_recalls


""" 
Constructs a dataframe to demonstrate the accuracy statistics
"""

def save_scores(male_precisions, male_recalls, female_precisions, female_recalls, \
    trans_female_precisions, trans_female_recalls, indir):
    """ 
    Save accuracy statistics to scores.csv in indir
    """
    # construct dataframe
    score_df = pd.DataFrame()
    score_df['male_precision'] = male_precisions
    score_df['male_recall'] = male_recalls
    score_df['female_precision'] = female_precisions
    score_df['female_recall'] = female_recalls
    score_df['trans_female_precision'] = trans_female_precisions
    score_df['trans_female_recall'] = trans_female_recalls
    # save
    score_df.to_csv(os.path.join(indir, "scores.csv"), index=None)