#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:38:54 2022

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
from utils_2L import *
import argparse

data_parser = argparse.ArgumentParser(description='Settings for generating data')

data_parser.add_argument('--directory', type=str, default='data/', help='Saving directory')
data_parser.add_argument('--num-data', type=int, default=100, help='Number of generated data')
data_parser.add_argument('--structured', type=bool, default=False, help='Structured or unstructured')
data_parser.add_argument('--PDE', type=str, default='Helmholtz', help='PDE problem')
data_parser.add_argument('--BC', type=str, default='Dirichlet', help='TBoundary conditions')
data_parser.add_argument('--ratio', type=tuple, default=(0.012, 0.03), help='Lower and upper bound for ratio')
data_parser.add_argument('--size-unstructured', type=tuple, default=(0.2, 0.5), help='Lower and upper bound for  unstructured size')
data_parser.add_argument('--size-structured', type=tuple, default=(10, 28), help='Lower and upper bound for structured size')
data_parser.add_argument('--hops', type=int, default=0, help='Learnable hops away from boundary')
data_parser.add_argument('--cut', type=int, default=1, help='RAS delta')

data_args = data_parser.parse_args()

path = data_args.directory+'TrainingGrids'

if not os.path.exists(path):
    os.makedirs(path)
size = np.random.uniform(low = data_args.size_unstructured[0], high = data_args.size_unstructured[1])
# ratio = np.random.uniform(low = data_args.ratio[0], high = data_args.ratio[1])

# old_g = rand_grid_gen(size, PDE = data_args.PDE)
lcmin = np.random.uniform(0.8, 0.9)

lcmax = np.random.uniform(0.9, 0.99)
n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
randomized = True if np.random.rand() < 0.4 else True
old_g = rand_grid_gen1(randomized = randomized, n = n, min_ = 0.03, min_sz = 0.6, 
              lcmin = lcmin, lcmax = lcmax, distmin = 0.01, distmax = 0.035, PDE = data_args.PDE)
# old_g = rand_grid_gen2(randomized = False, n = 12, var = 0.01, min_ = 0.0001, min_sz = 0.2, PDE = test_args.PDE)
# old_g = rand_grid_gen3(PDE = test_args.PDE)

num_node = old_g.num_nodes


ratio = 25*((old_g.num_nodes/600)**0.5)/old_g.num_nodes

grid =  Grid_PWA(old_g.A, old_g.mesh, max(2/old_g.num_nodes, ratio), hops = data_args.hops, 
                  cut=data_args.cut, h = 1, nu = 0, BC = data_args.BC)                

num_dom = grid.aggop[0].shape[-1]
 
   
torch.save(grid, path+"/grid.pth")
            
torch.save(data_args, path+"/data_config.pth")
            
