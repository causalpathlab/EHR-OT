import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import model
import transform as tran
import numpy as np
import os
import argparse
torch.set_num_threads(1)
import math
from read_data import ImageList



########################## Loading data ##########################
import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/OTTEHR/")
sys.path.append(f"/home/{user_id}/unbalanced_gromov_wasserstein/")
sys.path.append(f"/home/{user_id}/OTTEHR/competitors/")

from ast import literal_eval
import dnn
from mimic_common import *
import numpy as np
import os
import ot
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


group_1_count = 120
group_2_count = 100
iterations = 100
trans_metric = 'deepJDOT'
group_name = 'insurance'
group_1 = 'Self_Pay'
group_2 = 'Private'
selected_df = select_samples(admid_diagnosis_df, group_name, group_1, group_2, source_count=group_1_count, target_count=group_2_count)
source_traindata, source_trainlabel, target_traindata, target_trainlabel = gen_features_duration(selected_df, group_name, group_1, group_2)

########################## Loading data ##########################





use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def RSD(Feature_s, Feature_t, tradeoff2=0.1):
    u_s, s_s, v_s = torch.svd(Feature_s.t())
    u_t, s_t, v_t = torch.svd(Feature_t.t())
    p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return torch.norm(sinpa,1)+tradeoff2*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer

class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression,self).__init__()
        self.model_fc = model.Resnet18Fc()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer = nn.Sequential(self.classifier_layer,  nn.Sigmoid())
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
    def forward(self,x):
        feature = self.model_fc(x)
        outC= self.classifier_layer(feature)
        return(outC,feature)



Model_R = Model_Regression()
Model_R = Model_R.to(device)

Model_R.train(True)
criterion = {"regressor": nn.MSELoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, Model_R.classifier_layer.parameters()), "lr": 1}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
train_cross_loss = train_rsd_loss = train_total_loss = 0.0



# for param_group in optimizer.param_groups:
#     param_lr.append(param_group["lr"])
test_interval = 500
num_iter = 10002
for iter_num in range(1, num_iter + 1):
    Model_R.train(True)
    # optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
    #                              weight_decay=0.0005)
    optimizer.zero_grad()
    # if iter_num % len_source == 0:
    #     iter_source = iter(dset_loaders["train"])
    # if iter_num % len_target == 0:
    #     iter_target = iter(dset_loaders["val"])
    data_source = iter_source.next()
    data_target = iter_target.next()
    inputs_source, labels_source = data_source
    labels1 = labels_source[:,0]
    labels2 = labels_source[:,1]
    labels1 = labels1.unsqueeze(1)
    labels2 = labels2.unsqueeze(1)
    labels_source = torch.cat((labels1,labels2),dim=1)
    labels_source = labels_source.float()/39
    inputs_target, labels_target = data_target
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    inputs = inputs.to(device)
    labels = labels_source.to(device)
    inputs_s = inputs.narrow(0, 0, batch_size["train"])
    inputs_t = inputs.narrow(0, batch_size["train"], batch_size["train"])
    outC_s, feature_s = Model_R(inputs_s)
    outC_t, feature_t = Model_R(inputs_t)
    classifier_loss = criterion["regressor"](outC_s, labels)
    rsd_loss = RSD(feature_s,feature_t)
    tradeoff=0.1
    total_loss = classifier_loss + args.tradeoff*rsd_loss     # Combine the three losses
    total_loss.backward()
    optimizer.step()
    train_cross_loss += classifier_loss.item()
    train_rsd_loss += rsd_loss.item()
    train_total_loss += total_loss.item()
    if iter_num % 500 == 0:
        print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average RSD Loss: {:.4f};  Average Training Loss: {:.4f}".format(
            iter_num, train_cross_loss / float(test_interval), train_rsd_loss / float(test_interval),
            train_total_loss / float(test_interval)))
        train_cross_loss = train_rsd_loss = train_total_loss  = 0.0
    if (iter_num % test_interval) == 0:
        Model_R.eval()
        Regression_test(dset_loaders, Model_R.predict_layer)

