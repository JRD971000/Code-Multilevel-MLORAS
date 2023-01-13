#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:38:54 2022

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
from Unstructured import *
import scipy
from grids import *
import time
mpl.rcParams['figure.dpi'] = 300
from utils import *
import argparse


data_parser = argparse.ArgumentParser(description='Settings for generating data')

data_parser.add_argument('--directory', type=str, default='Data/new_data', help='Saving directory')
data_parser.add_argument('--num-data', type=int, default=10, help='Number of generated data')
data_parser.add_argument('--ratio', type=tuple, default=(0.012, 0.03), help='Lower and upper bound for ratio')
data_parser.add_argument('--size-unstructured', type=tuple, default=(0.2, 0.5), help='Lower and upper bound for  unstructured size')
data_parser.add_argument('--hops', type=int, default=1, help='Learnable hops away from boundary')
data_parser.add_argument('--cut', type=int, default=1, help='RAS delta')

data_args = data_parser.parse_args()


def generate_data(data_args, show_fig = False):
    
    path = data_args.directory
    
    if not os.path.exists(path):
        os.makedirs(path)
        

    for i in range(data_args.num_data):
        
        num_node = 0
        num_dom = 0

        while num_dom<2:
            
            size = np.random.uniform(low = data_args.size_unstructured[0], high = data_args.size_unstructured[1])

            lcmin = np.random.uniform(0.125, 0.14)

            lcmax = np.random.uniform(0.14, 0.16)
            n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
            randomized = True if np.random.rand() < 0.4 else True
            g = rand_grid_gen1(randomized = randomized, n = n, min_ = 0.03, min_sz = 0.6, 
                          lcmin = lcmin, lcmax = lcmax, distmin = 0.01, distmax = 0.035, PDE = 'Poisson')

            
            num_node = g.num_nodes
            ratio = 0.033 #25*((g.num_nodes/600)**0.5)/g.num_nodes

            grid =  Grid_PWA(g.A, g.mesh, max(2/g.num_nodes, ratio), hops = data_args.hops, 
                              cut=data_args.cut, h = 1, nu = 0, BC = 'Dirichlet') 
            num_dom = grid.aggop[0].shape[-1]
            
            
        print("grid number = ", i, ", number of nodes  ", num_node, ", number of domains = ", num_dom)
        
        if show_fig:
            grid.plot_agg(size = 1, labeling = False, w = 0.1,shade = 0.01)
            plt.title (f'Grid nodes = {grid.A.shape[0]}, subdomains = {num_dom}, nodes = {num_node}')
            plt.show()
   
        torch.save(grid, path+"/grid"+str(i)+".pth")
            
    torch.save(data_args, path+"/data_config.pth")
            
generate_data(data_args, show_fig = True)

