#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:21:11 2022

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

test_parser = argparse.ArgumentParser(description='Settings for training machine learning for ORAS')

test_parser.add_argument('--precond', type=bool, default=True, help='Test as a preconditioner')
test_parser.add_argument('--stationary', type=bool, default=True, help='Test as a stationary algorithm')
test_parser.add_argument('--structured', type=bool, default=False, help='Structured or unstructured')
test_parser.add_argument('--PDE', type=str, default='Helmholtz', help='PDE problem')
test_parser.add_argument('--BC', type=str, default='Dirichlet', help='TBoundary conditions')
test_parser.add_argument('--ratio', type=float, default=0.022, help='Lower and upper bound for ratio')
test_parser.add_argument('--TAGConv-k', type=int, default=2, help='TAGConv # of hops')
test_parser.add_argument('--epoch-num', type=int, default=3, help='Epoch number of the network being loaded')
test_parser.add_argument('--dim', type=int, default=128, help='Dimension of TAGConv filter')
test_parser.add_argument('--size-unstructured', type=float, default=0.1, help='Lower and upper bound for unstructured size')
test_parser.add_argument('--plot', type=bool, default=True, help='Plot the test grid')
test_parser.add_argument('--model_dir', type=str, default= 'Model-Grids-Helmholtz-Dirichlet', help='Directory for loading')
test_parser.add_argument('--size-structured', type=int, default=4, help='Lower and upper bound for structured size')
test_parser.add_argument('--hops', type=int, default=1, help='Learnable hops away from boundary')
test_parser.add_argument('--cut', type=int, default=1, help='RAS delta')

test_args = test_parser.parse_args()
num_lvl = 1

if num_lvl == 1:
    
    from utils_1L import *

if num_lvl == 2:
    
    from fgmres_2L import fgmres_2L
    from utils_2L import *

def test_fgmres(grid, dict_precs, list_test):
    
    n = grid.aggop[0].shape[0]
    
    x0 = np.random.random(grid.A.shape[0])
    x0 = x0/((grid.A@x0)**2).sum()**0.5
    
    b = np.zeros(grid.A.shape[0])
    
    dict_loss = {}
    
    for name in list_test:
        
        dict_loss[name] = []
        
        if num_lvl == 1:
            
            pyamg.krylov.fgmres(grid.A, b, x0=x0, tol=1e-12, 
                       restrt=None, maxiter=int(0.9*n),
                       M=dict_precs[name], callback=None, residuals=dict_loss[name])
            
        if num_lvl == 2:
            fgmres_2L(grid.A, b, x0=x0, tol=1e-12, 
                       restrt=None, maxiter=int(0.9*n),
                       M=dict_precs[name], grid = grid, callback=None, residuals=dict_loss[name])

    return dict_loss


def test_stats(grid, dict_stats, list_test):
    
    u = torch.rand(grid.x.shape[0],100).double()
    u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
    K = 10
    
    
    dict_enorm = {}
    
    for name in list_test:
        
        dict_enorm[name] = []
        
        if name == 'Jacobi':
            MJ = torch.tensor(np.diag(grid.A.diagonal()**(-1)))
            dict_enorm[name] = test_stationary(grid, None, precond_type = None, u = u, K = K, M = MJ)
        
        else:
            dict_enorm[name] = test_stationary(grid, None, precond_type = name, u = u, K = K, M = dict_stats[name]) 
            
    return dict_enorm



if __name__ =='__main__':
    
    if test_args.structured:
        
        ratio = test_args.ratio
        n = test_args.size_structured
        
        if test_args.BC == 'Dirichlet':
            old_g = structured(n, n, Neumann=False)
        else:
            old_g = structured(n, n, Neumann=True)  
            
        grid =  Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops, 
                          cut=test_args.cut, h = 1/(n+1), nu = 1)#, BC = 'Dirichlet')
        grid.aggop_gen(ratio = 0.1, cut = 1, node_agg = struct_agg_PWA(n,n,int(n/2),n))
        
    else:

        lcmin = 0.09#np.random.uniform(0.08, 0.09)
        lcmax = 0.11#np.random.uniform(0.14, 0.15)
        n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
        randomized = True if np.random.rand() < 0.4 else True
        old_g = rand_grid_gen1(randomized = randomized, n = n, min_ = 0.03, min_sz = 0.6, 
                      lcmin = lcmin, lcmax = lcmax, distmin = 0.01, distmax = 0.035, PDE = test_args.PDE)

        print(old_g.num_nodes)
        test_args.ratio = 12*((old_g.num_nodes/600)**0.5)/old_g.num_nodes
        grid =  Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops, 
                          cut=test_args.cut, h = 1, nu = 0, BC = test_args.BC)                
    
        
    the_types = 1
        
    if the_types == 0:
        save_dir = 'Data/test_results/const'
        grid.aggop_gen(0.02, 1)

    if the_types == 2:
        save_dir = 'Data/test_results/linear'
        grid.aggop_gen(12/grid.A.shape[0], 1)
    
    # grid = torch.load('data/Grids-Helmholtz-Dirichlet/grid90.pth')
    # gg = Grid_PWA(grid.A, grid.mesh, ratio = grid.ratio, hops=1)
    # gg.aggop_gen(grid.ratio, grid.cut, grid.aggop)
    # grid =  gg
    grid = torch.load('grid.pth')
    if test_args.plot:

        grid.plot_agg(size = 0.0, labeling = False, w = 0.1, shade=0.0008)

        plt.show()

    ts = time.time()

    # model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4, res = True, tf=True)
    
    # directory  = 'Models/'+test_args.model_dir+'/model_epoch'+str(test_args.epoch_num)+'.pth'
    
    
    model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4, res = True, tf=True)
    
    directory  = 'Models/'+test_args.model_dir+'/model_epoch_1_hop3.pth'
    
    # model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4, res = True, tf=False)

    # directory  = 'Models/Model-Grids-Helmholtz-Dirichlet/model_epoch_best7.pth'

    model.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
    
    list_test = ['RAS', 'ML_ORAS']
    list_label = {'RAS':'RAS','ML_ORAS': 'MLORAS'}
    dict_precs = {}
    dict_stats = {}
    n = grid.aggop[0].shape[0]
    
    x0 = np.random.random(grid.A.shape[0])
    x0 = x0/((grid.A@x0)**2).sum()**0.5
    
    data = grid.gdata
    data.edge_attr = data.edge_attr.float()
    model.eval()
    with torch.no_grad():
        out = model(data, grid)
    
    for name in list_test:
        
        if num_lvl == 1:
            M = preconditioner(grid, out, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
            
            dict_precs[name] = M.numpy()
            dict_stats[name] = M
        
        if num_lvl == 2:
            Mstat, Mprec = preconditioner(grid, out, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
            
            dict_precs[name] = Mprec.to_dense().detach().numpy()
            dict_stats[name] = Mstat
        
    
    if test_args.precond:
        
        dict_loss = test_fgmres(grid, dict_precs, list_test)
        for name in list_test:
            
            plt.plot(dict_loss[name][:-2], label = list_label[name], marker='.')
     

        plt.xlabel("fGMRES Iteration")
        plt.ylabel("Residual norm")
        plt.yscale('log')
        plt.legend()
        plt.title('GMRES convergence for '+str(int(grid.A.shape[0]))+'-node unstructured grid')
        plt.show()
        
    
    if test_args.stationary:

        dict_enorm = test_stats(grid, dict_stats, list_test)
        for name in list_test:
            
            plt.plot(dict_enorm[name], label = list_label[name], marker='.')
        

        tf = time.time()
        print('start-end = ', tf-ts)
        plt.xlabel("Iteration")
        plt.ylabel("error norm")
        plt.yscale('log')
        plt.ylim([1e-3, 1])
        plt.title('Stationary algorithm: Error norm for '+str(int(grid.A.shape[0]))+'-node unstructured grid')
        plt.legend()

        plt.show()






def test_on_list():

    list_grids = []
    for iii in range(1,10):
        list_grids.append(torch.load('Data/test_grids/grid'+str(iii)+'.pth'))
        
    for the_count in range(1,10):

        grid = list_grids[the_count-1]
        
        for the_types in range(3):
            
            if the_types == 0:
                save_dir = 'Data/test_results/const'
                grid.aggop_gen(0.02, 1)
                
            if the_types == 1:  
                save_dir = 'Data/test_results/square_root'
                grid.aggop_gen(12*((grid.A.shape[0]/600)**0.5)/grid.A.shape[0], 1)
                
            if the_types == 2:
                save_dir = 'Data/test_results/linear'
                grid.aggop_gen(12/grid.A.shape[0], 1)
                
            if test_args.plot:
        
                grid.plot_agg(size = 0.0, labeling = False, w = 0.1, shade=0.0008)
        
                plt.show()
        
            ts = time.time()
            model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4, res = True)
        
            directory  = 'Models/'+test_args.model_dir+'/model_epoch'+str(test_args.epoch_num)+'.pth'
            model.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
            
            list_test = ['RAS', 'ML_ORAS']
            dict_precs = {}
            dict_stats = {}
            n = grid.aggop[0].shape[0]
            
            x0 = np.random.random(grid.A.shape[0])
            x0 = x0/((grid.A@x0)**2).sum()**0.5
            
            data = grid.gdata
            data.edge_attr = data.edge_attr.float()
            out = model(data)
            
            for name in list_test:
                
                Mstat, Mprec = preconditioner(grid, out, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
                dict_precs[name] = Mprec.to_dense().detach().numpy()
                dict_stats[name] = Mstat
                
            
            if test_args.precond:
                
                dict_loss = test_fgmres(grid, dict_precs, list_test)
                torch.save(dict_loss, save_dir+'/dict_loss'+str(the_count)+'.pth')
                for name in list_test:
                    
                    plt.plot(dict_loss[name][:-2], label = name, marker='.')
             
        
                plt.xlabel("fGMRES Iteration")
                plt.ylabel("Residual norm")
                plt.yscale('log')
                plt.legend()
                plt.title('GMRES convergence for '+str(int(grid.A.shape[0]))+'-node unstructured grid')
                plt.show()
                
            
            if test_args.stationary:
        
                dict_enorm = test_stats(grid, dict_stats, list_test)
                torch.save(dict_enorm, save_dir+'/dict_enorm'+str(the_count)+'.pth')
                for name in list_test:
                    
                    plt.plot(dict_enorm[name], label = name, marker='.')
                
        
                tf = time.time()
                print('start-end = ', tf-ts)
                plt.xlabel("Iteration")
                plt.ylabel("error norm")
                plt.yscale('log')
                plt.ylim([1e-3, 1])
                plt.title('Stationary algorithm: Error norm for '+str(int(grid.A.shape[0]))+'-node unstructured grid')
                plt.legend()
        
                plt.show()
            
            

            