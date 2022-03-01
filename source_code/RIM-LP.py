import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
import copy
import sys
import os
import time
import argparse
import json
import numpy as np
import numpy.linalg as la
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from scipy.sparse import csgraph
from torch.backends import cudnn
from torch.optim import lr_scheduler
from utils import *
from graphConvolution import *

#hyperparameters
num_node = 2708
num_coreset = 20
num_class = 7
oracle_acc = 0.7
th = 0.05
batch_size = 5

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
#num_coreset = int((num_node-1500)*0.01)
hidden_size = 128
num_val = 500
num_test = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_reliable_score(similarity): #Eq7 Quntity Relability
    """
    This function implement Equation 7 in the paper
    """
    return (oracle_acc*similarity)/(oracle_acc*similarity+(1-oracle_acc)*(1-similarity)/(num_class-1))
    
def get_activated_node_dense(node,reliable_score,activated_node):
    """
    Description:
    This Function Implement Eq10. "th" used as threshold for finding
    activated (influenced) nodes by our specifice node as "node" 
    arguman. It outputs number of activated node as "Count" and list of 1,0 
    for indicating location (index) of activated node. In the first line
    of this function: 
    activated_vector=((adj_matrix2[node]*reliable_score)>th)+0, the
    adj_matrix2[node] is derivation of influenced node based on 
    input "node" after 2 iteration.

    argument:
    node <-- node that is labeling candidate
    reliable_score <-- reliability score of the nodes
    activated_node <-- the signature of influence quality
    """
    activated_vector=((adj_matrix2[node]*reliable_score)>th)+0 #Eq9 and Eq10
    #activated_vector=((adj_matrix2[node])>th)+0 #Just Using labeld influence not quality
    activated_vector=activated_vector*activated_node
    count=num_ones.dot(activated_vector)
    return count,activated_vector

def get_max_reliable_info_node_dense(idx_used,high_score_nodes,activated_node,train_class,labels):  
    """
    Description:
    This function work for selecting the best subset in each bath
    that maximize node influence quantity as quality. in every 
    implementation of this fucntion, it select the best and it's
    reliability and number od activated nodes by selecting this
    node.
    
    argumans:
    idx_used <-- idx_train
    high_score_nodes <-- index of availabel nodes for labaling (candidate)
    activated_node <-- activated_node (first iteration is ones vactor then use output of update_reliability)
    train_class <-- a dictionray, in keys we have classe and in values list of labaled nodes in previous batches
    labels <-- list labels
    """
    
    max_ral_node = 0
    max_activated_node = 0
    max_activated_num = 0   
    
    for node in high_score_nodes:
        reliable_score = oracle_acc
        
        activated_num,activated_node_tmp = get_activated_node_dense(node,reliable_score,activated_node)
        if activated_num > max_activated_num: #We wnat more activate nodes
            max_activated_num = activated_num #Num of Nodes that activated
            max_ral_node = node #Best for labaling
            max_activated_node = activated_node_tmp #Num of Nodes that activated    

    return max_ral_node,max_activated_node,max_activated_num


def update_reliability(idx_used,train_class,labels,num_node):
    """ 
    Description:
    This function in each batch update the reliability score of nodes.
    

    Argumans:
    idx_used <-- index of nodes that labeld in previos batches
    train_class <-- a dictionray, in keys we have classe and in values list of labaled nodes in previous batches 
    labels <-- label of nodes in tha hand of oracle (nois added)
    num_node <-- Number of nodes

    """

    activated_node = np.zeros(num_node)
    for node in idx_used:
        reliable_score = 0
        node_label = labels[node].item() 
        #-------------This Section Implement Eq 7 and 8 --------------
        if node_label in train_class:   #train_class is a dict that save index of nodes that has labels in 
            total_score = 0.0           #training based on their label.
            for tmp_node in train_class[node_label]: 
                total_score+=reliability_list[tmp_node] #Total Score: Reliablity score of all same labled node (dominator of Eq8)
            for tmp_node in train_class[node_label]:
                reliable_score+=reliability_list[tmp_node]*get_reliable_score(similarity_label[node][tmp_node]) #Numinator of Eq8 
            reliable_score = reliable_score/total_score #Eq8 Normilized reliablity score of candidate
        #--------------------------------------------------------
        else:
            reliable_score = oracle_acc 
        
        reliability_list[node]=reliable_score #reliabily_list is list contain reliability score (quality) of each labaled node
        activated_node+=((adj_matrix2[node]*reliable_score)>th)+0 #Eq11
        #activated_vector=((adj_matrix2[node])>th)+0 #Just Using labeld influence not quality
    return np.ones(num_node)-((activated_node>0)+0)

def my_cross_entropy(x_pred,x_traget):
    logged_x_pred = torch.log(x_pred)
    cost_value = -torch.sum(x_traget*logged_x_pred)/x_pred.size()[0]
    return cost_value

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)) #Sum row of adjecency 
    d_inv_sqrt = np.power(row_sum, -1.0).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break 
    return item 


def compute_cos_sim(vec_a,vec_b): #Cosine Similarity Measure
    return (vec_a.dot(vec_b.T))/(la.norm(vec_a)*la.norm(vec_b))
#read dataset

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid,bias=True)
        self.gc2 = GraphConvolution(nhid, nclass,bias=True)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

def find_maximum_acc_valid(acc_val,acc_test):
    max_val = 0
    max_test = 0
    for i in range(len(acc_val)):
        if acc_val[i]>max_val:
            max_val = acc_val[i]
            max_test = acc_test[i]
    return max_val,max_test
    

#read data
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="cora")

reliability_list = np.ones(num_node)
num_zeros = np.zeros(num_node)
num_ones = np.ones(num_node)
labels = list(labels.to(device))
idx_val = list(idx_val.to(device))
idx_test = list(idx_test.to(device))
idx_avaliable = list()
for i in range(num_node):
    if i not in idx_val and i not in idx_test:
        idx_avaliable.append(i)

# add noise
label_list=[]
prob_list = np.full((num_class,num_class),(1-oracle_acc)/(num_class-1)).tolist() #Error Probability

for i in range(num_class):
    label_list.append(i) #Just a list contain classes num : [0,1,...,Num_classes - 1]
    prob_list[i][i]=oracle_acc #Put each label trust accuracy to oracle_acc
for idx in idx_avaliable:
    labels[idx]=torch.tensor(random_pick(label_list,prob_list[labels[idx].item()])) #Add noise in a way that select label of nodes randomly
                                                                                    #By 0.3 probability of being wrong

#compute normalized distance

adj = aug_normalized_adjacency(adj)
adj_matrix = torch.FloatTensor(adj.todense()).to(device) #Change Format to Tensor
adj_matrix2 = torch.mm(adj_matrix,adj_matrix).to(device) #A*A used it as Eq6,Label Influence after 2 iteration
adj_matrix2 = np.array(adj_matrix2.to(device)) 
one_hot_label = F.one_hot(torch.Tensor(labels).to(torch.int64), num_classes=num_class).float() 
adj = sparse_mx_to_torch_sparse_tensor(adj).float().to(device)


print("node selection begin")
activated_node = np.ones(num_node) #An all one Vector [1,1,1,....,1] => cora -> shape (2708,)
idx_train = []
train_class = dict()
idx_avaliable_temp = copy.deepcopy(idx_avaliable) #idx_avaliable is list of training nodes index that isn't in test/val 
count = 0                                         #Also after each node selection batch, index of selected node removed


while True:
    """
    Iterate 140 (number of budget) and selecting best node
    for labaling in each iteration based on influence quantity and
    quality. In first batch we just used influence quantity.
    In each batch influence quality of candidate (non labaled node) 
    updated. Batch size = 5.
    """
    print("iteration:",count)
    max_ral_node,max_activated_node,max_activated_num = get_max_reliable_info_node_dense(idx_train,idx_avaliable_temp,activated_node,train_class,labels) 
    idx_train.append(max_ral_node) #add selected node for training
    idx_avaliable.remove(max_ral_node) #Remove Selected node form candidate
    idx_avaliable_temp.remove(max_ral_node) #Remove Selected node form candidate
    node_label = labels[max_ral_node].item() #Laibaling selected node 
    #------------------------------------
    """
    If in train_class thers is class label of selected node
    then append selected node index to it, if not, creat selected
    nodes label and then append it's index to it.
    """
    if node_label in train_class:
        train_class[node_label].append(max_ral_node)
    else:
        train_class[node_label]=list()
        train_class[node_label].append(max_ral_node)
    #-------------------------------------
    count += 1 #Flag of iteration
    
    #-------------------------------------
    """
    In each batch size update reliable quality of candidate
    """
    if count%batch_size == 0:
        
        one_hot_node_label_org = copy.deepcopy(one_hot_label)        
        one_hot_node_label = copy.deepcopy(one_hot_label)
        one_hot_node_label[idx_avaliable] = torch.zeros(num_class)
        one_hot_node_label[idx_train] = torch.zeros(num_class)
        one_hot_node_label[idx_test] = torch.zeros(num_class)
        
        for _ in range(2):
            one_hot_node_label = torch.mm(adj_matrix,one_hot_node_label)
            one_hot_node_label[idx_train] = one_hot_node_label_org[idx_train] 
        
        aax_label = np.array(one_hot_node_label.to(device))
        
        similarity_label = np.ones((num_node,num_node)) #Similarity Matrix between nodes
    
        for i in range(num_node-1):
            for j in range(i+1,num_node):
                similarity_label[i][j] = compute_cos_sim(aax_label[i],aax_label[j])
                similarity_label[j][i] = similarity_label[i][j] #Symetric 
        dis_range = np.max(similarity_label) - np.min(similarity_label)
        similarity_label = (similarity_label - np.min(similarity_label))/dis_range #Normilized Similarity    
        
        activated_node = update_reliability(idx_train,train_class,labels,num_node)

    """
    We have index of active nodes that activated by previous bahches
    , So we dont't want to activate them again.
    active_node -> Zeros mean activated node, Ones means they are not activated yet
    """
    activated_node = activated_node - max_activated_node
    #------------------------------------


    if count >= num_coreset or max_activated_num <= 0:
        """
        Stop iteration if we run out of budget
        """
        break



#------------------------------------
print("node selection end")
labels = torch.LongTensor(labels).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)
reliability_list = torch.FloatTensor(reliability_list).unsqueeze(1).to(device)



print('xxxxxxxxxx Evaluation begin xxxxxxxxxx')
t_total = time.time()
record = {}

predict_label = copy.deepcopy(one_hot_label)
predict_label[idx_avaliable] = torch.zeros(num_class)
predict_label[idx_val] = torch.zeros(num_class)
predict_label[idx_test] = torch.zeros(num_class)

for epoch in range(50):
    predict_label = torch.mm(adj_matrix,predict_label)
    predict_label[idx_train] = one_hot_label[idx_train].clone()

    acc_val = accuracy(predict_label[idx_val], labels[idx_val])
    acc_test = accuracy(predict_label[idx_test], labels[idx_test])
    record[acc_val.item()] = acc_test.item()


bit_list = sorted(record.keys())
bit_list.reverse()
#------------
max_val,max_test = find_maximum_acc_valid(list(record.keys()),list(record.values()))
print(f'Budget Numer: {num_coreset}')
print(f'Max Validation Accuracy: {max_val} and Max Test Accuracy: {max_test}')

#for key in bit_list:
#    value = record[key]
#    print(key,value)
print('xxxxxxxxxx Evaluation end xxxxxxxxxx')
