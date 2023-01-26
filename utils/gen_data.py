#!/usr/bin/env python3


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

data_parser.add_argument('--directory', type=str, default='Data/train_grids', help='Saving directory')
data_parser.add_argument('--num-data', type=int, default=10, help='Number of generated data')
data_parser.add_argument('--ratio', type=tuple, default=0.033, help='Coarsening ratio')
data_parser.add_argument('--hops', type=int, default=1, help='Learnable hops away from boundary')
data_parser.add_argument('--cut', type=int, default=1, help='RAS delta')
data_parser.add_argument('--show-fig', type=bool, default=False, help='Plot every generated grid')
data_parser.add_argument('--mesh-size', type=float, default=0.15, help='Mesh size')

data_args = data_parser.parse_args()


def generate_data(data_args, show_fig = False):
    
    path = data_args.directory
    
    if not os.path.exists(path):
        os.makedirs(path)
        

    for i in range(data_args.num_data):
        
        num_node = 0
        num_dom = 0

        while num_dom<2:
            
            
            n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
            g = rand_grid_gen1(randomized = True, n = n, min_ = 0.03, min_sz = 0.6, 
                          lcmin = data_args.mesh_size, lcmax = data_args.mesh_size, distmin = 0.05, distmax = 0.035, PDE = 'Poisson')

            
            num_node = g.num_nodes

            grid =  Grid_PWA(g.A, g.mesh, max(2/g.num_nodes, data_args.ratio), hops = data_args.hops, 
                              cut=data_args.cut, h = 1, nu = 0, BC = 'Dirichlet') 
            num_dom = grid.aggop[0].shape[-1]
            
        print("grid number = ", i, ", number of nodes  ", num_node, ", number of domains = ", num_dom)
        
        if show_fig:
            grid.plot_agg(size = 1, labeling = False, w = 0.1,shade = 0.01)
            plt.title (f'Grid nodes = {grid.A.shape[0]}, subdomains = {num_dom}, nodes = {num_node}')
            plt.show()
   
        torch.save(grid, path+"/grid"+str(i)+".pth")
            
    torch.save(data_args, path+"/data_config.pth")
            
generate_data(data_args, show_fig = data_args.show_fig)

