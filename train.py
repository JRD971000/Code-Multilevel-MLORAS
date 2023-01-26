#!/usr/bin/env python3

import sys
sys.path.append('utils')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import os
import os.path
from grids import *
import torch as T
import copy
import random
from Unstructured import *
import scipy
from grids import *
from utils import *
import argparse
from mggnn import *
from lloyd_gunet import *

train_parser = argparse.ArgumentParser(description='Settings for training machine learning for ORAS')

train_parser.add_argument('--num-epoch', type=int, default=10, help='Number of training epochs')
train_parser.add_argument('--mini-batch-size', type=int, default=10, help='Coarsening ratio for aggregation')
train_parser.add_argument('--lr', type=float, default= 5e-4, help='Learning rate')
train_parser.add_argument('--TAGConv-k', type=int, default=2, help='TAGConv # of hops')
train_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
train_parser.add_argument('--data-set', type=str, default='Data/train_grids', help='Directory of the training data')
train_parser.add_argument('--K', type=int, default=10, help='Number of iterations in the loss function')
train_parser.add_argument('--GNN', type=str, default='MG-GNN', help='MG-GNN or Graph-Unet')

train_args = train_parser.parse_args()




if __name__ == "__main__":
    
        
    path = 'Models/new-train'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    list_grids = []
    
    num_data = sum((len(f) for _, _, f in os.walk(train_args.data_set)))-1

    for i in range(num_data):

        g = torch.load(train_args.data_set+"/grid"+str(i)+".pth")
        list_grids.append(g)

    print('Finished Uploading Training Data')
    
    if train_args.GNN == 'MG-GNN':
        model = MGGNN(lvl=2, dim_embed=128, num_layers=4, K=train_args.TAGConv_k, ratio=0.2, lr=train_args.lr)
    elif train_args.GNN == 'Graph-Unet':
        model = lloyd_gunet(2, 4, 128, K = 2, ratio = 0.2, lr = train_args.lr)
    else:
        raise ValueError("Select GNN architecture between MG-GNN and Graph-Unet")
    
    print('Number of parameters: ',sum(p.numel() for p in model.parameters()))

    epoch_loss_list = []
    all_indices = np.arange(num_data)
    model.optimizer.zero_grad()
    
    current_best_loss = 10**12


    epoch_loss = 0

    for epoch in range(train_args.num_epoch):

        loss = 0
        np.random.shuffle(all_indices)
        mbs = train_args.mini_batch_size
        print("Epoch = ", epoch)
        print("-----------------")
        for count in range(int(np.ceil((num_data)/mbs))):

            batch_idxs = all_indices[count*mbs:min((count+1)*mbs, num_data)]

            for i in batch_idxs:

                grid = list_grids[i]
                data = grid.gdata.to(device)
                data.edge_attr = data.edge_attr.float()
                output = model.forward(data, grid, train = True)
                
                u = torch.rand(grid.x.shape[0],100).double().to(device)
                u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
  
                current_loss = stationary_max(grid, output, u = u, K = train_args.K, precond_type='ML_ORAS')

                loss += current_loss
                
            loss.backward()

            model.optimizer.step()

            epoch_loss += loss.item()
            print ("batch = ", count, "loss = ", loss.item())
            model.optimizer.zero_grad() 
            
            loss = 0
        
        epoch_loss_list.append(epoch_loss)
        print('** Epoch loss is = ', epoch_loss)
        
        if epoch_loss < current_best_loss:
            torch.save(model.state_dict(), path+"/model_epoch_best.pth")
            torch.save(model.state_dict(), path+"/model_epoch"+str(epoch)+".pth")   
            current_best_loss = epoch_loss
            torch.save(epoch_loss_list, path+"/loss_list.pth")
        epoch_loss = 0

        
        
        print("-----------------")
        

    torch.save(train_args, path+"/training_config.pth")
    torch.save(epoch_loss_list, path+"/loss_list.pth")
    
    plt.plot(epoch_loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss vs. Iteration')
    plt.show()  
    