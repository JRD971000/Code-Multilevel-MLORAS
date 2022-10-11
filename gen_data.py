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
data_parser.add_argument('--num-data', type=int, default=1000, help='Number of generated data')
data_parser.add_argument('--structured', type=bool, default=False, help='Structured or unstructured')
data_parser.add_argument('--PDE', type=str, default='Helmholtz', help='PDE problem')
data_parser.add_argument('--BC', type=str, default='Dirichlet', help='TBoundary conditions')
data_parser.add_argument('--ratio', type=tuple, default=(0.012, 0.03), help='Lower and upper bound for ratio')
data_parser.add_argument('--size-unstructured', type=tuple, default=(0.2, 0.5), help='Lower and upper bound for  unstructured size')
data_parser.add_argument('--size-structured', type=tuple, default=(10, 28), help='Lower and upper bound for structured size')
data_parser.add_argument('--hops', type=int, default=1, help='Learnable hops away from boundary')
data_parser.add_argument('--cut', type=int, default=1, help='RAS delta')

data_args = data_parser.parse_args()


def generate_data(data_args, show_fig = False):
    
    path = data_args.directory+'data-800-1k'
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    if data_args.structured:
        for i in range(data_args.num_data):
            
            num_dom = 0
            while num_dom < 2:
                
                ratio = np.random.uniform(low = data_args.ratio[0], high = data_args.ratio[1])
                n = 10+2*i#np.random.choice(np.arange(data_args.size_structured[0], data_args.size_structured[1]+1))
                    
                if data_args.BC == 'Dirichlet':
                    old_g = structured(n, n, Neumann=False)
                else:
                    old_g = structured(n, n, Neumann=True)  
                    
                grid =  Grid_PWA(old_g.A, old_g.mesh, max(2/old_g.num_nodes, ratio), hops = data_args.hops, 
                                  cut=data_args.cut, h = 1/(n+1), nu = 1)#, BC = 'Dirichlet')
                grid.aggop_gen(ratio = 0.1, cut = data_args.cut, node_agg = struct_agg_PWA(n,n,int(n/2),n))

                num_dom = grid.aggop[0].shape[-1]
            
            print(i, "  ", grid.A.shape[0])
            
            if show_fig:
                grid.plot_agg(size = 10, fsize = 3)
                plt.show()
       
            torch.save(grid, path+"/grid"+str(i)+".pth")
            
    else:
        for i in range(data_args.num_data):
            
            # num_dom = 0
            num_node = 0

            while  num_node < 800 or num_node > 1000:
                
                size = np.random.uniform(low = data_args.size_unstructured[0], high = data_args.size_unstructured[1])
                # ratio = np.random.uniform(low = data_args.ratio[0], high = data_args.ratio[1])

                # old_g = rand_grid_gen(size, PDE = data_args.PDE)
                lcmin = np.random.uniform(0.125, 0.14)

                lcmax = np.random.uniform(0.14, 0.16)
                n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
                randomized = True if np.random.rand() < 0.4 else True
                old_g = rand_grid_gen1(randomized = randomized, n = n, min_ = 0.03, min_sz = 0.6, 
                              lcmin = lcmin, lcmax = lcmax, distmin = 0.01, distmax = 0.035, PDE = data_args.PDE)
                # old_g = rand_grid_gen2(randomized = False, n = 12, var = 0.01, min_ = 0.0001, min_sz = 0.2, PDE = test_args.PDE)
                # old_g = rand_grid_gen3(PDE = test_args.PDE)
                
                num_node = old_g.num_nodes
                t1 = time.time()
                
                ratio = 0.033 #25*((old_g.num_nodes/600)**0.5)/old_g.num_nodes
                
            grid =  Grid_PWA(old_g.A, old_g.mesh, max(2/old_g.num_nodes, ratio), hops = data_args.hops, 
                              cut=data_args.cut, h = 1, nu = 0, BC = data_args.BC)                
            t2 = time.time()
 
            
            print(i, "  ", num_node)
            
            if show_fig:
                grid.plot_agg(size = 1, labeling = False, w = 0.1,shade = 0.003)
                plt.title (f'Grid nodes = {grid.A.shape[0]}, subdomains = {num_dom}, ratio = {np.round(num_dom/grid.A.shape[0],2)}')
                plt.show()
       
            torch.save(grid, path+"/grid"+str(i)+".pth")
            
    torch.save(data_args, path+"/data_config.pth")
            
# generate_data(data_args, show_fig = False)


for i in range(17):
    print(i)
    g = torch.load('Data/test_grids/grid'+str(i)+'.pth')
    grid = Grid_PWA(g.A, g.mesh, 0.2, hops = 1, 
                      cut=1, h = 1, nu = 0, BC = 'Dirichlet')    
    ratio = max(25*((g.A.shape[0]/600)**0.5)/g.A.shape[0], 2/g.A.shape[0])
    grid.aggop_gen(ratio, 1)
    torch.save(grid, 'Data/test_grids_2/grid'+str(i)+'.pth')