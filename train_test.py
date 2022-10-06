#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:51:54 2022

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
    
train_parser = argparse.ArgumentParser(description='Settings for training machine learning for ORAS')

train_parser.add_argument('--num-epoch', type=int, default=100, help='Number of training epochs')
train_parser.add_argument('--mini-batch-size', type=int, default=1, help='Coarsening ratio for aggregation')
train_parser.add_argument('--lr', type=float, default= 1e-4, help='Learning rate')
train_parser.add_argument('--TAGConv-k', type=int, default=2, help='TAGConv # of hops')
train_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
train_parser.add_argument('--data-set', type=str, default='TrainingGrids', help='Directory of the training data')
train_parser.add_argument('--K', type=int, default=4, help='Number of iterations in the loss function')

train_args = train_parser.parse_args()

    
print(device)
print("********")
print(train_args)
