#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 00:07:48 2022

@author: alitaghibakhshi
"""

# import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import os
import os.path
from grids import *
import sys
import torch as T
import copy
import random

from Unstructured import *
import scipy
from grids import *
import time
# mpl.rcParams['figure.dpi'] = 300
from ST_CYR import *

import argparse

from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# indiv_loss = torch.load('Models/Model-gnn/indiv_loss_nolog.pth')
# for i in range(10):
#     plt.plot([indiv_loss['epoch '+str(j)]['grid '+str(i)] for j in range(100)], label = 'Grid '+str(i))
#     # plt.plot([indiv_loss['epoch '+str(j)]['grid '+str(i)] for j in range(100)], label = 'Grid '+str(i), marker='.')

# plt.yscale('log')
# plt.legend()
# plt.show()
# sys.exit()
print(device)
print('********')
train_parser = argparse.ArgumentParser(description='Settings for training machine learning for ORAS')

train_parser.add_argument('--num-epoch', type=int, default=200, help='Number of training epochs')
train_parser.add_argument('--mini-batch-size', type=int, default=1, help='Coarsening ratio for aggregation')
train_parser.add_argument('--lr', type=float, default= 1e-4, help='Learning rate')
train_parser.add_argument('--TAGConv-k', type=int, default=6, help='TAGConv # of hops')
train_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
train_parser.add_argument('--data-set', type=str, default='data-800-1k', help='Directory of the training data')
train_parser.add_argument('--K', type=int, default=10, help='Number of iterations in the loss function')

train_args = train_parser.parse_args()


num_lvl = 2
NN_debug = False

if NN_debug:
    from NN_experiment import *
else:
    from NeuralNet import *
    from hgnn import *

if num_lvl == 1:
    
    from utils_1L import *

if num_lvl == 2:
    
    from fgmres_2L import fgmres_2L
    from utils_2L import *
    


if __name__ == "__main__":
    
        
    path = 'Models/Model-gnn'#+train_args.data_set
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    list_grids = []
    
    num_data = sum((len(f) for _, _, f in os.walk('Data/'+train_args.data_set)))-1
    num_data = 1
    for i in range(num_data):
        g = torch.load('grid2.pth')#'Data/'+train_args.data_set+"/grid"+str(i)+".pth")
    
        list_grids.append(g)
    
    # gg = torch.load('Data/trainingdata/0.pth')
    # # gg = Grid_PWA(g.A, g.mesh, ratio = 0.3, hops=-1)
    # # gg.aggop_gen(g.ratio, g.cut, g.aggop)
    # list_grids.append(gg)
    # g = torch.load('Data/'+train_args.data_set+"/grid.pth")
    # list_grids = [g]
    print('Finished Uploading Training Data')
    # model = FC_test(g.A.shape[0], g.gmask.nonzero()[0].shape[0], 128, lr = 1e-4)
    model = HGNN(lvl=2, dim_embed=128, num_layers=6, K= train_args.TAGConv_k, ratio=0.05, lr=train_args.lr)
    # model = mloras_net(dim = 128, K = train_args.TAGConv_k, num_res = 8, num_convs = 4, lr = train_args.lr, res = True, tf=False)
    # model.load_state_dict(torch.load("/Users/alitaghibakhshi/PycharmProjects/ML_OSM/OSM_ML/All_Models/Models_for_Grids_pretrain3/model_epoch99.pth"))
    model.to(device)
    
    epoch_loss_list = []

    # u = torch.rand(grid.x.shape[0],5000).double()
    # u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
    
    all_indices = np.arange(num_data)
    model.optimizer.zero_grad()
    
    current_best_loss = 10**12
    
    epoch_loss = 0
    # indiv_loss = {}
    # probs = np.zeros_like(all_indices)
    for epoch in range(train_args.num_epoch):
        # indiv_loss['epoch '+str(epoch)] = {}
        # t0 = time.time()
        loss = 0
        np.random.shuffle(all_indices)
        mbs = train_args.mini_batch_size
        print("Epoch = ", epoch)
        print("-----------------")
        for count in range(int(np.ceil((num_data)/mbs))):

            batch_idxs = all_indices[count*mbs:min((count+1)*mbs, num_data)]
            # if epoch != 0:
            #     batch_idxs = np.random.choice(all_indices, size = min((count+1)*mbs, num_data) - count*mbs, p = np.exp(probs)/sum(np.exp(probs)))

            for i in batch_idxs:

                grid = list_grids[i]
                data = grid.gdata.to(device)
                data.edge_attr = data.edge_attr.float()
                output = model.forward(data, grid)
                
                u = torch.rand(grid.x.shape[0],100).double().to(device)
                u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
                
                current_loss = stationary_max(grid, output, u = u, K = train_args.K, precond_type='ML_ORAS')
                
                # indiv_loss['epoch '+str(epoch)]['grid '+str(i)] = current_loss.item()
                # probs[i] = current_loss.item()
                loss += current_loss
                
            # t1 = time.time()
            loss.backward()
            # t2 = time.time()
            model.optimizer.step()

            epoch_loss += loss.item()
            print ("batch = ", count, "loss = ", loss.item())
            model.optimizer.zero_grad() 
            
            # print(f't10 = {t1-t0}\n')
            # print(f't21 = {t2-t1}\n')

            # sys.exit()
            loss = 0
        
        epoch_loss_list.append(epoch_loss)
        print('** Epoch loss is = ', epoch_loss)
        
        if epoch_loss < current_best_loss:
            torch.save(model.state_dict(), path+"/model_epoch_best.pth")
            current_best_loss = epoch_loss
            torch.save(epoch_loss_list, path+"/loss_list.pth")
        epoch_loss = 0

        
        
        print("-----------------")
        
    # torch.save(model.state_dict(), path+"/model_epoch"+str(epoch)+".pth")   
    # torch.save(indiv_loss, path+"/indiv_loss_nolog.pth")
    torch.save(train_args, path+"/training_config.pth")
    torch.save(epoch_loss_list, path+"/loss_list.pth")
    
    # plt.plot(epoch_loss_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.yscale('log')
    # plt.title('Loss vs. Iteration')
    # plt.show()  
    