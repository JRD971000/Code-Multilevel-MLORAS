#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:51:54 2022

@author: alitaghibakhshi
"""

import matplotlib.pyplot as plt
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
from NeuralNet import *
from Unstructured import *
import scipy
from grids import *
import time
mpl.rcParams['figure.dpi'] = 300
from ST_CYR import *

import argparse

from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch

train_parser = argparse.ArgumentParser(description='Settings for training machine learning for ORAS')

train_parser.add_argument('--num-epoch', type=int, default=100, help='Number of training epochs')
train_parser.add_argument('--mini-batch-size', type=int, default=1, help='Coarsening ratio for aggregation')
train_parser.add_argument('--lr', type=float, default= 1e-5, help='Learning rate')
train_parser.add_argument('--TAGConv-k', type=int, default=2, help='TAGConv # of hops')
train_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
train_parser.add_argument('--data-set', type=str, default='Grids-Helmholtz-Dirichlet', help='Directory of the training data')
train_parser.add_argument('--K', type=int, default=4, help='Number of iterations in the loss function')

train_args = train_parser.parse_args()


num_lvl = 2

if num_lvl == 1:
    
    from utils_1L import *

if num_lvl == 2:
    
    from fgmres_2L import fgmres_2L
    from utils_2L import *
    
    
if __name__ == "__main__":
    
        
    path = 'Models/Model-'+train_args.data_set
    
    if not os.path.exists(path):
        os.makedirs(path)

    list_grids = []
    
    num_data = sum((len(f) for _, _, f in os.walk('data/'+train_args.data_set)))-1
    num_data = 1
    for i in range(num_data):
        g = torch.load('data/'+train_args.data_set+"/grid"+str(i)+".pth")
        gg = Grid_PWA(g.A, g.mesh, ratio = g.ratio, hops=-1)
        gg.aggop_gen(g.ratio, g.cut, g.aggop)
        list_grids.append(gg)
        
    # g = torch.load('grid.pth')
    # gg = Grid_PWA(g.A, g.mesh, ratio = 0.3, hops=-1)
    # gg.aggop_gen(g.ratio, g.cut, g.aggop)
    # list_grids.append(gg)
    
    print('Finished Uploading Training Data')
    model = mloras_net(dim = train_args.dim, K = train_args.TAGConv_k, num_res = 8, num_convs = 4, lr = train_args.lr, res = True, tf=True)
    # model.load_state_dict(torch.load("/Users/alitaghibakhshi/PycharmProjects/ML_OSM/OSM_ML/All_Models/Models_for_Grids_pretrain3/model_epoch99.pth"))
    gg.plot_agg(size = 5, fsize =1)
    plt.show()
    loss_list = []

    # u = torch.rand(grid.x.shape[0],5000).double()
    # u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
    
    all_indices = np.arange(num_data)
    model.optimizer.zero_grad()
    for epoch in range(train_args.num_epoch):
        loss = 0
        np.random.shuffle(all_indices)
        mbs = train_args.mini_batch_size

        for count in range(int(np.ceil((num_data)/mbs))):
            batch_idxs = all_indices[count*mbs:min((count+1)*mbs, num_data)]
            data =  Batch.from_data_list([list_grids[idx].gdata for idx in batch_idxs])
            
            row, col = data.edge_index
            edge_batch = data.batch[row]
            
            grid = list_grids[batch_idxs[0]]
            data.edge_attr = data.edge_attr.float()
            batch_output = model.forward(data, grid)
            
            
            for i in range(data.ptr.shape[0]-1):
                start = edge_batch.tolist().index(i)
                if i == data.ptr.shape[0]-2:
                    out   = batch_output[start:]
                else:
                    stop  = edge_batch.tolist().index(i+1)
                    out   = batch_output[start:stop]
                                    
                grid = list_grids[batch_idxs[i]]
                u = torch.rand(grid.x.shape[0],5000).double()
                u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
                
                loss += stationary_max(grid, out, u = u, K = train_args.K, precond_type='ML_ORAS')
            

            loss_list.append(loss.item())
            print (epoch, loss)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
        
            loss = 0
            
        torch.save(model.state_dict(), path+"/model_epoch_best"+str(epoch)+".pth")
        
    torch.save(train_args, path+"/training_config.pth")
    torch.save(loss_list, path+"/loss_list.pth")
    
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss vs. Iteration')
    plt.show()  
    