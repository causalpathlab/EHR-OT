""" 
Common functions for synthetic datasets
"""
import sys
sys.path.append("/home/wanxinli/deep_patient")

import numpy as np
import matplotlib.pyplot as plt
import ot
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

""" 
Transport female representations to male representations
"""

def trans_female2male(male_reps, female_reps, type="balanced"):
    """ 
    Optimal transport (without entropy regularization) female representations \
        to male representations

    :param str type: balanced or unbalanced
    :returns: transported female representations
    """
    trans_female_reps = None
    if type == "balanced":
        print("enter balanced")
        ot_emd = ot.da.SinkhornTransport(reg_e=1e-1)
        ot_emd.fit(Xs=female_reps, Xt=male_reps)
        trans_female_reps = ot_emd.transform(Xs=female_reps)

    elif type == "unbalanced":
        reg = 0.005
        reg_m_kl = 0.5
        n = female_reps.shape[0]

        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

        M = ot.dist(female_reps, male_reps)
        M /= M.max()

        coupling = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m_kl)
        print(female_reps.shape)
        print(coupling.shape)
        trans_female_reps = np.matmul(coupling, female_reps)

    print("before return is:", trans_female_reps)
    return trans_female_reps


""" 
Caculate result statistics for binary labels
"""

def cal_stats_binary(male_reps, male_labels, female_reps, female_labels, \
    trans_female_reps, model_func):
    """ 
    Calculate accuracy statistics based on logistic regression between the \
        patient representations and label labels
    This function is for binary labels

    :param function model_func: the function to model the relationship between \
        representations and reponse
    
    :returns: using the male model,\
        - accuracy for male/female/transported female
        - precision for male/female/transported female
        - recall for male/female/transported female
        - f1 for male/female/transported female
            
    """
    # fit the model
    male_model = model_func()
    male_model.fit(male_reps, male_labels)

    # calculate the stats
    male_pred_labels = male_model.predict(male_reps)
    male_accuracy = accuracy_score(male_labels, male_pred_labels)
    male_precision = precision_score(male_labels, male_pred_labels)
    male_recall = recall_score(male_labels, male_pred_labels)
    male_f1 = f1_score(male_labels, male_pred_labels, average="weighted")

    female_pred_labels = male_model.predict(female_reps)
    print("female_labels are:", female_labels)
    print("female_pred_labels are:", female_pred_labels)
    female_accuracy = accuracy_score(female_labels, female_pred_labels)
    female_precision = precision_score(female_labels, female_pred_labels)
    female_recall = recall_score(female_labels, female_pred_labels)
    female_f1 = f1_score(female_labels, female_pred_labels, average="weighted")

    trans_female_pred_labels = male_model.predict(trans_female_reps)
    print("trans_female_pred_labels are:", trans_female_pred_labels)
    trans_female_accuracy = accuracy_score(female_labels, trans_female_pred_labels)
    trans_female_precision = precision_score(female_labels, trans_female_pred_labels)
    trans_female_recall = recall_score(female_labels, trans_female_pred_labels)
    trans_female_f1 = f1_score(female_labels, trans_female_pred_labels, average="weighted")


    return male_accuracy, male_precision, male_recall, male_f1, \
        female_accuracy, female_precision, female_recall, female_f1, \
        trans_female_accuracy, trans_female_precision, trans_female_recall, trans_female_f1


""" 
Wrap up everything for binary labels
"""

def entire_proc_binary(sim_func, custom_train_reps, model_func):
    """ 
    Executes the entire procedure including
        - generate male sequences, male labels, female sequences and female labels
        - generate male representations and female representations
        - transport female representations to male representations
        - train logistic regression model using male representations and male expires
        - calculate accuracy statistics for males, females and transported females

    :param function sim_func: simulation function
    :param function custom_train_reps: customized deep patient function for training representations
    :param function model_func: the function to model the relationship bewteen representations and response
    :returns: the accuracy scores
    """
    male_seqs, male_labels, female_seqs, female_labels = sim_func(num_patient = 50)
    male_reps, female_reps = custom_train_reps(male_seqs, female_seqs)
    trans_female_reps = trans_female2male(male_reps, female_reps)
    
    male_accuracy, male_precision, male_recall, male_f1, \
        female_accuracy, female_precision, female_recall, female_f1, \
        trans_female_accuracy, trans_female_precision, trans_female_recall, trans_female_f1 = \
        cal_stats_binary(male_reps, male_labels, female_reps, female_labels, trans_female_reps, model_func)
    return male_accuracy, male_precision, male_recall, male_f1, \
        female_accuracy, female_precision, female_recall, female_f1, \
        trans_female_accuracy, trans_female_precision, trans_female_recall, trans_female_f1
    


""" 
Run entire procedure on multiple simulations and print accuracy statistics, \
    for binary labels
"""

def run_proc_multi(sim_func, custom_train_reps, model_func, n_times = 100):
    """ 
    Run the entire procedure (entire_proc) multiple times (default 100 times), \
        for binary labels

    :param function model_func: the function to model the relationship between representations and responses
    :param bool filter: whether to filter out female accuracies > 0.7

    :returns: vectors of accuracy statistics of multiple rounds
    """
    
    print("calling run_proc_multi")
    male_accuracies = []
    male_precisions = [] 
    male_recalls = [] 
    male_f1s = []
    female_accuracies = []
    female_precisions = []
    female_recalls = [] 
    female_f1s = []
    trans_female_accuracies = []
    trans_female_precisions = []
    trans_female_recalls = []
    trans_female_f1s = []

    for _ in range(n_times):
        # init accuracies
        male_accuracy = None
        male_precision = None
        male_recall = None
        male_f1 = None
        female_accuracy = None
        female_precision = None
        female_recall = None
        female_f1 = None
        trans_female_accuracy = None
        trans_female_precision = None
        trans_female_recall = None
        trans_female_f1 = None

        try:
            print("enter try")
            male_accuracy, male_precision, male_recall, male_f1, \
            female_accuracy, female_precision, female_recall, female_f1, \
            trans_female_accuracy, trans_female_precision, trans_female_recall, trans_female_f1 = \
                    entire_proc_binary(sim_func, custom_train_reps, model_func)
                    
            # print("printing out accuracies")
            # print(male_accuracy, male_precision, male_recall, \
            # female_accuracy, female_precision, female_recall, \
            # trans_female_accuracy, trans_female_precision, trans_female_recall)
        except Exception: # most likely only one label is generated for the examples
            print("exception 1")
            continue

        # if domain 2 data performs better using the model trained by domain 1 data, \
        # there is no need to transport
        if male_accuracy <= female_accuracy: 
            print("exception 2")
            continue

        # denominator cannot be 0
        min_deno = 0.001
        male_accuracy = max(male_accuracy, min_deno)
        male_precision = max(male_precision, min_deno)
        male_recall = max(male_recall, min_deno)
        male_f1 = max(male_f1, min_deno)
        female_accuracy = max(female_accuracy, min_deno)
        female_precision = max(female_precision, min_deno)
        female_recall = max(female_recall, min_deno)
        female_f1 = max(female_f1, min_deno)
        trans_female_accuracy = max(trans_female_accuracy, min_deno)
        trans_female_precision = max(trans_female_precision, min_deno)
        trans_female_recall = max(trans_female_recall, min_deno)
        trans_female_f1 = max(trans_female_f1, min_deno)

        male_accuracies.append(male_accuracy)
        male_precisions.append(male_precision)
        male_recalls.append(male_recall)
        male_f1s.append(male_f1)
        female_accuracies.append(female_accuracy)
        female_precisions.append(female_precision)
        female_recalls.append(female_recall)
        female_f1s.append(female_f1)
        trans_female_accuracies.append(trans_female_accuracy)
        trans_female_precisions.append(trans_female_precision)
        trans_female_recalls.append(trans_female_recall) 
        trans_female_f1s.append(trans_female_f1)
    return male_accuracies, male_precisions, male_recalls, male_f1s, \
        female_accuracies, female_precisions, female_recalls, female_f1s, \
        trans_female_accuracies, trans_female_precisions, trans_female_recalls, trans_female_f1s


""" 
Constructs a dataframe to demonstrate the accuracy statistics for binary labels
"""

def save_scores(male_accuracies, male_precisions, male_recalls, male_f1s, \
        female_accuracies, female_precisions, female_recalls, female_f1s, \
        trans_female_accuracies, trans_female_precisions, trans_female_recalls, trans_female_f1s, file_path):
    """ 
    Save accuracy statistics to file path
    """
    # construct dataframe
    score_df = pd.DataFrame()
    score_df['male_accuracy'] = male_accuracies
    score_df['male_precision'] = male_precisions
    score_df['male_recall'] = male_recalls
    score_df['male_f1'] = male_f1s
    score_df['female_accuracy'] = female_accuracies
    score_df['female_precision'] = female_precisions
    score_df['female_recall'] = female_recalls
    score_df['female_f1'] = female_f1s
    score_df['trans_female_accuracy'] = trans_female_accuracies
    score_df['trans_female_precision'] = trans_female_precisions
    score_df['trans_female_recall'] = trans_female_recalls
    score_df['trans_female_f1'] = trans_female_f1s
    # save
    score_df.to_csv(file_path, index=None, header=True)


""" 
Box plot of simulation result statistics
"""

def box_plot(scores_path, filter = True):
    """ 
    Box plot of the scores in score dataframe stored in scores_path for binary labels. \
        Specifically, we plot the box plots of 
        - precision/recall of female over accuracy/precision/recall of male
        - precision/recall of transported female over accuracy/precision/recall of male
        - precision/recall of transported female over accuracy/precision/recall of female

    :param str scores_path: the path to scores.csv
    :param bool filter: filter out scores where female accuracy is greater than > 0.7 (small room for improvement)
    """

    scores_df = pd.read_csv(scores_path, index_col=None, header=0)

    male_accuracy = scores_df['male_accuracy']
    male_f1 = scores_df['male_f1']

    female_accuracy = scores_df['female_accuracy']
    female_f1 = scores_df['female_f1']

    trans_female_accuracy = scores_df['trans_female_accuracy']
    trans_female_f1 = scores_df['trans_female_f1']

    if filter:
        delete_indices = []
        high_acc_thres = 0.7
        for i in range(len(female_accuracy)):
            if female_accuracy[i] > high_acc_thres:
                delete_indices.append(i)
        print("delete_indices is:", delete_indices)
        print("male_accuracy is:", male_accuracy)
        male_accuracy = np.delete(list(male_accuracy), delete_indices)
        male_f1 = np.delete(list(male_f1), delete_indices)
        female_accuracy = np.delete(list(female_accuracy), delete_indices)
        female_f1 = np.delete(list(female_f1), delete_indices)
        trans_female_accuracy = np.delete(list(trans_female_accuracy), delete_indices)
        trans_female_f1 = np.delete(list(trans_female_f1), delete_indices)
    

    trans_female_female_accuracy_incre =  [i - j for i, j in zip(trans_female_accuracy, female_accuracy)]
    print(trans_female_female_accuracy_incre)
    trans_female_female_f1_incre =  [i - j for i, j in zip(trans_female_f1, female_f1)]
    print("number of stats is:", len(trans_female_female_accuracy_incre))
    print("number of 0 in incre is:", trans_female_female_accuracy_incre.count(0))
    print("number of elements > 0 is:", np.sum(np.array(trans_female_female_accuracy_incre) > 0, axis=0))
    print("number of elements < 0 is:", np.sum(np.array(trans_female_female_accuracy_incre) < 0, axis=0))
    print("average trans female to female accuracy increment is:", np.mean(trans_female_female_accuracy_incre))
    print("median trans female to female accuracy increment is:", np.median(trans_female_female_accuracy_incre))
    print("average trans female to female accuracy f1 is:", np.mean(trans_female_female_f1_incre ))
    print("median trans female to female accuracy f1 is:", np.median(trans_female_female_f1_incre))

    fig = plt.figure(figsize=(16,16))
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'fuchsia'}

    # female to male accuracy
    female_male_accuracy = [i / j for i, j in zip(female_accuracy, male_accuracy)]

    # transported female to male accuracy
    trans_female_male_accuracy = [i / j for i, j in zip(trans_female_accuracy, male_accuracy)]

    # transported female to female accuracy
    trans_female_female_accuracy = [i / j for i, j in zip(trans_female_accuracy, female_accuracy)]
    print("average trans female to female accuracy is:", np.mean(trans_female_female_accuracy))
    print("median trans female to female accuracy is:", np.median(trans_female_female_accuracy))


    # female to male accuracy
    female_male_f1 = [i / j for i, j in zip(female_f1, male_f1)]

    # transported female to male accuracy
    trans_female_male_f1 = [i / j for i, j in zip(trans_female_f1, male_f1)]

    # transported female to female accuracy
    trans_female_female_f1 = [i / j for i, j in zip(trans_female_f1, female_f1)]
    print("average trans female to female f1 is:", np.mean(trans_female_female_f1))
    print("median trans female to female f1 is:", np.median(trans_female_female_f1))


    plt.subplot(3, 3, 1)
    plt.boxplot(female_male_accuracy, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.title("female accuracy to \n male accuracy")

    
    plt.subplot(3, 3, 2)
    plt.boxplot(trans_female_male_accuracy, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.title("transported female \n accuracy to \n male accuracy")

    
    plt.subplot(3, 3, 3)
    plt.boxplot(trans_female_female_accuracy, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.axhline(y = 1, color = 'b', linestyle = '-')
    plt.title("transported female \n accuracy to \n female accuracy")

    plt.subplot(3, 3, 4)
    plt.boxplot(female_male_f1, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.title("female precision to \n male f1")

    
    plt.subplot(3, 3, 5)
    plt.boxplot(trans_female_male_f1, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.title("transported female \n precision to \n male f1")

    
    plt.subplot(3, 3, 6)
    plt.boxplot(trans_female_female_f1, flierprops=flierprops)
    # plt.ylim(y_min, y_max)
    plt.axhline(y = 1, color = 'b', linestyle = '-')
    plt.title("transported female \n precision to \n female f1")

    


    plt.tight_layout()
    plt.show()


""" 
Histogram plot of simulation result statistics
"""
def hist_plot(scores_path, filter = True):
    """ 
    histogram plot of the scores in score dataframe stored in scores_path for binary labels. \
        Specifically, we plot the box plots of 
        - precision/recall of female over accuracy/precision/recall of male
        - precision/recall of transported female over accuracy/precision/recall of male
        - precision/recall of transported female over accuracy/precision/recall of female

    :param str scores_path: the path to scores.csv
    :param bool filter: filter out scores where female accuracy is greater than > 0.7 (small room for improvement)
    """

    scores_df = pd.read_csv(scores_path, index_col=None, header=0)

    male_accuracy = scores_df['male_accuracy']
    male_f1 = scores_df['male_f1']

    female_accuracy = scores_df['female_accuracy']
    female_f1 = scores_df['female_f1']

    trans_female_accuracy = scores_df['trans_female_accuracy']
    trans_female_f1 = scores_df['trans_female_f1']

    if filter:
        delete_indices = []
        high_acc_thres = 0.7
        for i in range(len(female_accuracy)):
            if female_accuracy[i] > high_acc_thres:
                delete_indices.append(i)
        male_accuracy = np.delete(list(male_accuracy), delete_indices)
        male_f1 = np.delete(list(male_f1), delete_indices)
        female_accuracy = np.delete(list(female_accuracy), delete_indices)
        female_f1 = np.delete(list(female_f1), delete_indices)
        trans_female_accuracy = np.delete(list(trans_female_accuracy), delete_indices)
        trans_female_f1 = np.delete(list(trans_female_f1), delete_indices)
    

    trans_female_female_accuracy_incre =  [i - j for i, j in zip(trans_female_accuracy, female_accuracy)]
    trans_female_female_f1_incre =  [i - j for i, j in zip(trans_female_f1, female_f1)]

    print("average trans female to female accuracy increment is '{:.1%}".format(np.mean(trans_female_female_accuracy_incre)))
    print("median trans female to female accuracy increment is '{:.1%}".format(np.median(trans_female_female_accuracy_incre)))
    print("average trans female to female accuracy f1 is '{:.1%}".format(np.mean(trans_female_female_f1_incre)))
    print("median trans female to female accuracy f1 is '{:.1%}".format(np.median(trans_female_female_f1_incre)))

    fig = plt.figure(figsize=(16,16))
    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'fuchsia'}

    # female to male accuracy
    female_male_accuracy = [i / j for i, j in zip(female_accuracy, male_accuracy)]

    # transported female to male accuracy
    trans_female_male_accuracy = [i / j for i, j in zip(trans_female_accuracy, male_accuracy)]

    # transported female to female accuracy
    trans_female_female_accuracy = [i / j for i, j in zip(trans_female_accuracy, female_accuracy)]


    # female to male accuracy
    female_male_f1 = [i / j for i, j in zip(female_f1, male_f1)]

    # transported female to male accuracy
    trans_female_male_f1 = [i / j for i, j in zip(trans_female_f1, male_f1)]

    # transported female to female accuracy
    trans_female_female_f1 = [i / j for i, j in zip(trans_female_f1, female_f1)]

    bin_width = 0.01
    plt.subplot(3, 3, 1)
    plt.hist(trans_female_female_accuracy_incre, \
        bins=np.arange(min(trans_female_female_accuracy_incre), max(trans_female_female_accuracy_incre) + bin_width, bin_width))
    plt.title("trans female to female accuracy increment histogram")

    
    plt.subplot(3, 3, 2)
    plt.hist(trans_female_female_f1_incre , \
        bins=np.arange(min(trans_female_female_f1_incre), max(trans_female_female_f1_incre) + bin_width, bin_width))
    plt.title("trans female to female f1 increment histogram")

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

