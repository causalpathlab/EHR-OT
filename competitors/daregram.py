import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/OTTEHR/")
sys.path.append(f"/home/{user_id}/unbalanced_gromov_wasserstein/")
sys.path.append(f"/home/{user_id}/OTTEHR/competitors/")

from common import *
from mimic_common import *
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import transform as tran
import numpy as np
import os
import argparse


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch DARE-GRAM experiment')
parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
parser.add_argument('--lr', type=float, default=0.1,
                        help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001,
                        help='learning rate decay')
parser.add_argument('--tradeoff_angle', type=float, default=0.05,
                        help='tradeoff for angle alignment')
parser.add_argument('--tradeoff_scale', type=float, default=0.001,
                        help='tradeoff for scale alignment')
parser.add_argument('--threshold', type=float, default=0.9,
                        help='threshold for the pseudo inverse')
parser.add_argument('--batch', type=int, default=36,
                        help='batch size')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)



# set dataset
batch_size = {"source": args.batch, "target": args.batch}

# Define dimensions
num_samples_source = 1000  # Adjust based on your source needs
num_samples_target = 1200    # Adjust based on your target needs
num_features = 512        # Total number of features for each example, adjust as needed

# Create synthetic data using numpy
np.random.seed(args.seed)  # Ensure reproducibility
source_data = np.random.rand(num_samples_source, num_features).astype(np.float32)
source_labels = np.random.rand(num_samples_source).astype(np.float32)
target_data = np.random.rand(num_samples_target, num_features).astype(np.float32)
target_labels = np.random.rand(num_samples_target).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
source_data = torch.tensor(source_data)
source_labels = torch.tensor(source_labels)
target_data = torch.tensor(target_data)
target_labels = torch.tensor(target_labels)




# Create datasets
source_dataset = PreparedDataset(source_data, source_labels)
target_dataset = PreparedDataset(target_data, target_labels)

# Create data loaders
batch_size = {"source": args.batch, "target": args.batch}
dset_loaders = {
    "source": torch.utils.data.DataLoader(source_dataset, batch_size=batch_size["source"], shuffle=True, num_workers=4),
    "target": torch.utils.data.DataLoader(target_dataset, batch_size=batch_size["target"], shuffle=False, num_workers=4)
}



def daregram_loss(H1, H2):  

    def add_noise(feature, eps=1e-7):
        # Normalize the matrix: subtract the mean and divide by the standard deviation of each feature
        mean = feature.mean(dim=0, keepdim=True)
        std = feature.std(dim=0, keepdim=True)
        matrix_norm = (feature - mean) / (std + eps)  # Adding a small value to prevent division by zero

        # Add small Gaussian noise to the matrix to improve stability
        noise = torch.randn_like(matrix_norm) * eps
        noisy_feature = matrix_norm + noise
        return noisy_feature  
    
    b,p = H1.shape

    print("H1 is:", H1)
    A = torch.cat((torch.ones(b,1), H1), 1)
    B = torch.cat((torch.ones(b,1), H2), 1)

    print("A is:", A)

    cov_A = (A.t()@A)
    cov_B = (B.t()@B) 
    
    print("cov_A is:", cov_A)


    _,L_A,_ = torch.svd(add_noise(cov_A))
    _,L_B,_ = torch.svd(add_noise(cov_B))
    
    eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

    if(eigen_A[1]>args.threshold):
        T = eigen_A[1].detach()
    else:
        T = args.threshold
        
    index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

    if(eigen_B[1]>args.threshold):
        T = eigen_B[1].detach()
    else:
        T = args.threshold

    index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
    
    k = max(index_A, index_B)[0]

    A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
    B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
    
    cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
    cos = torch.dist(torch.ones((p+1)),(cos_sim(A,B)),p=1)/(p+1)
    
    return args.tradeoff_angle*(cos) + args.tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer

# class Regression(nn.Module):
#     def __init__(self):
#         super(Regression,self).__init__()
#         self.model_fc = model.Resnet18Fc()
#         self.classifier_layer = nn.Linear(512, 3)
#         self.classifier_layer.weight.data.normal_(0, 0.01)
#         self.classifier_layer.bias.data.fill_(0.0)
#         self.classifier_layer = nn.Sequential(self.classifier_layer,  nn.Sigmoid())
#         self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)

#     def forward(self,x):
#         feature = self.model_fc(x)
#         outC= self.classifier_layer(feature)
#         return(outC, feature)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super(LinearRegressionModel, self).__init__()
        # Feature extraction layers
        self.extraction = nn.Linear(input_features, hidden_units)  # First hidden layer

        # Linear regression output layer
        self.output = nn.Linear(hidden_units, output_features)  # Output layer

    def forward(self, x):
        feature = self.extraction(x)
        x = self.output(feature)
        return feature, x


n_components = 50
Model_R = LinearRegressionModel(input_features=source_data.shape[1], output_features=1, hidden_units=n_components)
# Model_R = Model_R.to(device)

Model_R.train(True)
criterion = {"regressor": nn.MSELoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.extraction.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, Model_R.output.parameters()), "lr": 1}]

optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)


len_source = len(dset_loaders["source"]) - 1
len_target = len(dset_loaders["target"]) - 1
param_lr = []
iter_source = iter(dset_loaders["source"])
iter_target = iter(dset_loaders["target"])

for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])

test_interval = 5
num_iter = 100

train_regression_loss = train_dare_gram_loss = train_total_loss =  0.0
print(args)

for iter_num in range(1, num_iter + 1):
    print("iter_num is:", iter_num)
    Model_R.train(True)
    optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                 weight_decay=0.0005)
    optimizer.zero_grad()
    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["source"])
    if iter_num % len_target == 0:
        iter_target = iter(dset_loaders["target"])

    data_source = iter_source.next()
    data_target = iter_target.next()

    inputs_source, labels_source = data_source
    inputs_target, labels_target = data_target
    print("print input data shape")
    print(inputs_source.shape, labels_source.shape)
    print(inputs_target.shape, labels_target.shape)


    feature_s, outC_s,  = Model_R(inputs_source)
    feature_t, outC_t = Model_R(inputs_target)


    regression_loss = criterion["regressor"](outC_s, labels_source)
    dare_gram_loss = daregram_loss(feature_s,feature_t)
    print("daregram_loss is:", dare_gram_loss)

    total_loss = regression_loss + dare_gram_loss

    total_loss.backward()

    # Clip graidents
    clip_value = 1
    torch.nn.utils.clip_grad_norm_(Model_R.parameters(), clip_value)

    optimizer.step()

    train_regression_loss += regression_loss.item()
    train_dare_gram_loss += dare_gram_loss.item()
    train_total_loss += total_loss.item()
    if iter_num % test_interval == 0:
        print("Iter {:05d}, Average MSE Loss: {:.4f}; Average DARE-GRAM Loss: {:.4f}; Average Training Loss: {:.4f}".format(
            iter_num, train_regression_loss / float(test_interval), train_dare_gram_loss / float(test_interval), train_total_loss / float(test_interval)))
        train_regression_loss = train_dare_gram_loss = train_total_loss =  0.0


# Evaluate the model with training data
with torch.no_grad():  # We don't need gradients in the testing phase

    # Similarly for testing data
    print("target_data shape is:", target_data.shape)
    _, target_pred_labels = Model_R(target_data)
    target_pred_labels = target_pred_labels.data.numpy()
    print("target_pred_labels shape is:", target_pred_labels.shape)
    target_labels = target_labels.data.numpy()
    
    test_RMSE = np.sqrt(np.mean((target_pred_labels- target_labels) ** 2))
    test_MAE = np.mean(np.abs(target_pred_labels - target_labels))
    print("test RMSE is:", test_RMSE, "test_MAE is:", test_MAE)

      