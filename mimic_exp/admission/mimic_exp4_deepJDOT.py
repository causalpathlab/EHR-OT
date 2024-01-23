import sys
sys.path.append("/home/wanxinli/OTTEHR/")
sys.path.append("/home/wanxinli/unbalanced_gromov_wasserstein/")

from ast import literal_eval
from mimic_common import *
import numpy as np
import os
import ot
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def l2_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def optimal_transport_plan(x_s, x_t, alpha, reg=0.1, reg_m=1):

    source_measure = np.ones((x_s.shape[0],))/x_s.shape[0]
    target_measure = np.ones((x_t.shape[0],))/x_t.shape[0]

    M = ot.dist(x_s, x_t, metric='euclidean')
    M /= M.max()
    gamma = ot.unbalanced.sinkhorn_unbalanced(source_measure, target_measure, M, reg, reg_m)

    return gamma * alpha


def linear_regression(x, params):
    x_with_intercept = np.column_stack((x, np.ones(x.shape[0])))
    return np.matmul(x_with_intercept, params)


def pairwise_operator(vector1, vector2, operator):
    """
    Calculate pairwise operator between two vectors.

    Parameters:
    - vector1 (list): First input vector
    - vector2 (list): Second input vector
    - operator (function): Pairwise operator function (e.g., add, subtract, multiply, divide)

    Returns:
    - result_matrix (list of lists): Matrix containing pairwise operation results
    """

    result_matrix = [[operator(vector1[i], vector2[j]) for j in range(len(vector2))] for i in range(len(vector1))]
    
    return np.array(result_matrix)


def objective_function(params, x_s, x_t, y_s, alpha, lambda_t, n_components):
    n_s, n_t = len(x_s), len(x_t)
    g_params = params[:x_s.shape[1] * n_components].reshape(-1, n_components)
    f_params = params[x_s.shape[1] * n_components:]
    
    # Embedding functions g
    g_x_s = np.matmul(x_s, g_params)
    g_x_t = np.matmul(x_t, g_params)
    
    # Linear regression function f
    y_pred = linear_regression(g_x_t, f_params)
    
    # Calculate the optimal transport plan
    gamma = optimal_transport_plan(x_s, x_t, alpha)
    
    # Calculate the objective function value
    result_matrix_1 = pairwise_operator(x_s, x_t, lambda x, y: np.linalg.norm(x-y)**2)
    result_matrix_2 = pairwise_operator(y_s, y_pred, lambda x, y: np.linalg.norm(x-y)**2)

    loss_term_1 = np.sum(np.matmul(gamma, np.transpose(result_matrix_1)))
    loss_term_2 = np.sum(np.matmul(gamma, np.transpose(lambda_t * result_matrix_2)))
    
    objective_value = loss_term_1 + loss_term_2
    print("loss is:", objective_value)
    
    return objective_value


def train_models(x_s, x_t, y_s, alpha, lambda_t, n_components):
    # Initial guess for parameters (you may want to adjust this based on your problem)
    initial_params = np.random.rand((x_s.shape[1] * n_components) + n_components+1) # +1 for the intercept in linear regression
    
    # Define the optimization problem
    objective_function_partial = lambda params: objective_function(params, x_s, x_t, y_s, alpha, lambda_t, n_components)
    
    # Use a numerical optimization method to find optimal parameters
    result = minimize(objective_function_partial, initial_params, method='BFGS')
    
    # Extract optimal parameters for g and f
    optimal_params = result.x
    optimal_g_params = optimal_params[:x_s.shape[1] * n_components].reshape(-1, n_components)
    optimal_f_params = optimal_params[x_s.shape[1] * n_components:]
    
    return optimal_g_params, optimal_f_params

def calc_stats(x_t, y_t, g_params, f_params):
    """ 
    Calculates the MAE and RMSE of the dataset 
    """
    g_x_t = np.matmul(x_t, g_params)
    y_pred = linear_regression(g_x_t, f_params)
    rmse = np.sqrt(mean_squared_error(y_t, y_pred))
    mae = np.mean(np.abs(y_t - y_pred))
    return rmse, mae



output_dir = os.path.join(os.path.expanduser("~"), f"OTTEHR/outputs/mimic")
print(f"Will save outputs to {output_dir}")

n_components = 50

suffix = None
group_name = 'marital_status'
group_1 =   'MARRIED'
group_2 = 'SEPARATED'
group_1_count = 120
group_2_count = 100

""" 
Read in the original dataframe
"""
admid_diagnosis_df = pd.read_csv(os.path.join(output_dir, "ADMID_DIAGNOSIS.csv"), index_col=0, header=0, converters={'ICD codes': literal_eval})
print(admid_diagnosis_df)


selected_df = select_df_cts(admid_diagnosis_df, group_name, group_1, group_2, source_count=group_1_count, target_count=group_2_count)

source_features, source_labels, target_features, target_labels = gen_features_duration(selected_df, group_name, group_1, group_2)

alpha = 1                    # Example value for alpha
lambda_t = 1                 # Example value for lambda_t
n_components = 50               # Example value for the reduced dimensions, consistent with other experiments

g_params, f_params = train_models(source_features, target_features, source_labels, alpha, lambda_t, n_components)
print("g_params is:", g_params, "f_params is:", f_params)
rmse, mae = calc_stats(source_features, source_labels, g_params, f_params)
print(f"rmse is: {rmse}, mae is: {mae}")
