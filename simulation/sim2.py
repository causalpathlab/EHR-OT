#!/usr/bin/env python
# coding: utf-8

# Simulation goals
# 
# * Continuous response variable
# 
# * Different feature distributions for different domains
# 
# 
# Simulation
# 
# * $D$: total number of features
# 
# * $d_{1}$: number of features with higher frequency in a domain 1
# 
# * $d_{2}$: number of features with higher frequency in a domain 2
# 
# 
# * $d_{1} \sim \operatorname{Unif}(0, \lfloor D/4 \rfloor)$
# 
# * $d_{2} \sim \operatorname{Unif}(0, \lfloor D/4 \rfloor)$ ($d_{1} + d_{2} \le D$)
# 
# * $k \in [D]$ for indexing feature
# 
# * Let $\Delta_{r} = \{j \in [D]: \textrm{feature } j \textrm{ is more frequent in a domain } r \}$
# 
# * Sample $\Delta_{1} \subseteq [D]$ such that $|\Delta_{1}| = d_{1}$ 
# 
# * Sample $\Delta_{2} \subseteq [D]\backslash \Delta_{1}$ such that $|\Delta_{2}| = d_{2}$
# 
# * Let $\alpha_{1} \overset{\Delta}{=} \left( \alpha_{11}, \ldots, \alpha_{1D} \right)$ be a feature frequency vector for a domain 1
# 
# * Let $\alpha_{2} \overset{\Delta}{=} \left( \alpha_{21}, \ldots, \alpha_{2D} \right)$ be a feature frequency vector for a domain 2
# 
# 
# * For each $k \in [D]$: 
# 
#     * If $k \in \Delta_{1}$, $\alpha_{1k} > \alpha_{2k}$
# 
#     * If $k \in \Delta_{2}$, $\alpha_{2k} > \alpha_{1k}$
# 
#     * Otherwise $\alpha_{1k} = \alpha_{2k}$
# 
# 
# * Sample $\rho_{1} \sim \operatorname{Dir}(\alpha_{1})$
# 
# * Sample $\rho_{2} \sim \operatorname{Dir}(\alpha_{2})$
# 
# * Sample a code-specific contribution to mortality: $W_{k} \sim  \max\{ \mathcal{N}\!\left(0,1\right), 0\}$
# 
# * For each patient $i$ in a domain $r$
# 
#     * $\tilde{X}_{i} \sim \operatorname{Multi}(n_{i}; \rho_{r})$ where $\tilde{X}_{i}$ is a vector of counts for each diagnosis code/feature, $n_i = 3k$.
# 
# 
# * For each patient $i$ in a domain $r$
# 
# 	* $\lambda_i = \sum_{k} (W_{k} X_{ik} + b)$ 
# 
#     * Sample $Y_i = \operatorname{Poisson}(\lambda_i)$

# In[1]:


import sys
sys.path.append("/home/wanxinli/deep_patient/synthetic_exp")

from common import *
from deep_patient.sda import SDA
from math import floor, exp
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import dirichlet
import ot
from numpy.random import poisson
import pandas as pd
from random import randint
from scipy import sparse
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm

base_dir = "/home/wanxinli/deep_patient"


# In[2]:


""" 
Simulation scheme
"""

def simulate(D, d_1, d_2, num_patient):
    """ 
    Simulate features and labels for domain 1 and domain 2
    :param int D:  total number of features
    :param int d_1: number of features with higher frequency in domain 1
    :param int d_2: number of features with higher frequency in domain 2
    :param int num_patient: number of patients in each domain

    Variables in the implementation are consistent with the variables in the scheme

    TODO: reconsider the choice of alpha_1 and alpha_2

    :return
        list[list[int]] domain 1 features
        list[int] domain 1 labels
        list[list[int]] domain 2 features
        list[int] domain 2 labels
    """

    d_1 = randint(0, floor(0.25*D))
    d_2 = randint(0, floor(0.25*D))
    delta_1 = np.random.choice(size = d_1, a = range(1, D+1), replace=False)
    remaining_set = list(set(list(range(1, D+1)))-set(delta_1))
    delta_2 = np.random.choice(size = d_1, a = remaining_set, replace=False)
    
    # We set the proportions of d_1 codes, d_2 codes, and (D-d_1-d_2) codes to be 2:1:1.5
    unit_1 = 1/(0.5*d_1-0.5*d_2+1.5*D)
    alpha_1 = [2*unit_1]*d_1
    alpha_1.extend([unit_1]*d_2)
    alpha_1.extend([1.5*unit_1]*(D-d_1-d_2))

    # We set the proportions of d_1 codes, d_2 codes, and (D-d_1-d_2) codes to be 1:2:1.5
    unit_2 = 1/(-0.5*d_1+0.5*d_2+1.5*D)
    alpha_2 = [2*unit_2]*d_1
    alpha_2.extend([unit_2]*d_2)
    alpha_2.extend([1.5*unit_2]*(D-d_1-d_2))    

    def gen_feature_vector_label(alpha):
        """ 
        Generate feature vectors and labels
        :param list[float] alpha: concentration parameteres for the dirichlet distribution
        """

        def sigmoid(x):
            return 1 / (1 + exp(-x))

        rho = dirichlet(alpha=alpha, size=1)[0]
        W = np.random.normal(size=D)
        W = [max(0, W_k) for W_k in W] # only sample positive weights
        X = []
        Y = []
        b = 0
        all_sum = []

        for _ in range(num_patient):
            X_i = np.random.multinomial(3*len(rho), rho)
            for k in range(len(X_i)):
                if X_i[k] > 0:
                    X_i[k] = 1 # dominant effect
            X.append(X_i)
            lambda_i= np.sum(np.multiply(W, X_i))
            Y_i = poisson(lam = lambda_i)
            Y.append(Y_i)
      
        return X, Y, W, b
    
    def feature_vector_to_feature(feature_vectors):
        """ 
        Convert feature vectors to features
        :param list[list[int]]: feature vectors consisting of indicators

        Returns
            - features consisting of actual codes
        """
        features = []
        for feature_vector in feature_vectors:
            features.append([i for i, e in enumerate(feature_vector) if e != 0])
        return features
    
    def pad_features(features_list):
        """ 
        Pad features to the same length (maximum length of the original features)\
            in each domain by -1
        """
        max_len = 0
        for features in features_list:
            max_len = max(max_len, len(features))

        for i in range(len(features_list)):
            features_list[i] += [-1] * (max_len - len(features_list[i]))
        return features_list



    feature_vector_1, label_1, W_1, b_1 = gen_feature_vector_label(alpha_1)
    feature_1 = pad_features(feature_vector_to_feature(feature_vector_1))
    feature_vector_2, label_2, W_2, b_2 = gen_feature_vector_label(alpha_2)
    feature_2 = pad_features(feature_vector_to_feature(feature_vector_2))
    return np.array(feature_1), np.array(label_1), np.array(feature_2), np.array(label_2)


# In[3]:


def simulation_wrapper(num_patient):
    D = 20
    d_1 = 5
    d_2 = 5
    return simulate(D, d_1, d_2, num_patient)


# In[4]:


"""
Train deep patient model and generate representations for targets and sources
"""

def custom_train_reps(target_seqs, source_seqs):
    """ 
    Customized training algorithm for generating target representations and source representations
    
    :returns: target representations, source representations
    """

    # customized parameters
    nhidden = 3
    nlayer = 1

    # for targets
    # initiate the model
    target_sda = SDA(target_seqs.shape[1],
                nhidden=nhidden,
                nlayer=nlayer,
                param={
        'epochs': 100,
        'batch_size': 5,
        'corrupt_lvl': 0.05
    })

    # train the model
    target_sda.train(target_seqs)

    # apply the mode
    target_reps = target_sda.apply(target_seqs)

    # for sources
    # initiate the model
    source_sda = SDA(source_seqs.shape[1],
                nhidden=nhidden,
                nlayer=nlayer,
                param={
        'epochs': 100,
        'batch_size': 5,
        'corrupt_lvl': 0.05
    })

    # train the model
    source_sda.train(source_seqs)

    # apply the mode
    source_reps = source_sda.apply(source_seqs)
    return target_reps, source_reps


# In[5]:


target_seqs, target_labels, source_seqs, source_labels = simulation_wrapper(50)
target_reps, source_reps = custom_train_reps(target_seqs, source_seqs)


# In[ ]:


target_maes, target_mses, target_rmses,  source_maes, source_mses, source_rmses,     trans_source_maes, trans_source_mses, trans_source_rmses =     run_proc_multi_cts(simulation_wrapper, custom_train_reps, linear_model.LinearRegression, n_times = 100)


# In[ ]:


score_path = "../outputs/sim2_lr_scores.csv"
save_scores_cts(target_maes, target_mses, target_rmses,  source_maes, source_mses, source_rmses,         trans_source_maes, trans_source_mses, trans_source_rmses, score_path)


# In[ ]:


""" 
Smaller is better
"""

hist_plot_cts(score_path)


# In[ ]:




