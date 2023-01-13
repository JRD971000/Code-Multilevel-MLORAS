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
from Unstructured import *
import scipy
from grids import *
import time
from fgmres import *
from utils import *
from lloyd_gunet import *
from mggnn import *
mpl.rcParams['figure.dpi'] = 300
import argparse

test_parser = argparse.ArgumentParser(description='Settings for training machine learning for 2-level MLORAS')

test_parser.add_argument('--precond', type=bool, default=True, help='Test as a preconditioner')
test_parser.add_argument('--stationary', type=bool, default=True, help='Test as a stationary algorithm')
test_parser.add_argument('--plot', type=bool, default=False, help='Plot the test grid')
test_parser.add_argument('--model-dir', type=str, default= 'Models/model_trained.pth', help='Model directory')
test_parser.add_argument('--GNN', type=str, default= 'MG-GNN', help='MG-GNN or Graph-Unet')
test_parser.add_argument('--data-dir', type=str, default= 'Data/test_grids', help='Test data directory')
test_parser.add_argument('--data-index', type=int, default= 7, help='Index of the test grid')

test_args = test_parser.parse_args()

def torch_2_scipy_sparse(A):
    
    data = A.coalesce().values()
    row = A.coalesce().indices()[0]
    col = A.coalesce().indices()[1]
    out = scipy.sparse.csr_matrix((data, (row, col)), shape=(A.shape[0], A.shape[1]))
    
    return out

def test_fgmres(grid, dict_precs, list_test):
    
    n = grid.aggop[0].shape[0]
    
    x0 = np.random.random(grid.A.shape[0])
    x0 = x0/((grid.A@x0)**2).sum()**0.5
    
    b = np.zeros(grid.A.shape[0])
    
    dict_loss = {}

    for name in list_test:
        
        dict_loss[name] = []
    
        fgmres_2L(grid.A, b, x0=x0, tol=1e-12, 
                   restrt=None, maxiter=int(0.9*n),
                   M=dict_precs[name], grid = grid, callback=None, residuals=dict_loss[name])

    return dict_loss


def test_stats(grid, dict_stats, list_test):
    
    u = torch.rand(grid.A.shape[0],25).double()
    u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
    K = 200
    
    
    dict_enorm = {}
    dict_vects = {}
    for name in list_test:
        
        dict_enorm[name] = []
        dict_vects[name] = []
        
        dict_enorm[name], dict_vects[name] = test_stationary(grid, dict_stats[name], precond_type = name, u = u, K = K, M = dict_stats[name][0]) 
            
    return dict_enorm, dict_vects


if __name__ =='__main__':

        grid = torch.load(test_args.data_dir+'/grid'+str(test_args.data_index)+'.pth')

        if test_args.plot:
    
            grid.plot_agg(size = 0.08, labeling = False, w = 0.1, shade=0.007)
            plt.show()

                
        if test_args.GNN == 'MG-GNN':
            model = MGGNN(lvl=2, dim_embed=128, num_layers=4, K=2, ratio=0.2, lr=1e-4)
        elif test_args.GNN == 'Graph-Unet':
            model = lloyd_gunet(2, 4, 128, K = 2, ratio = 0.2, lr = 1e-4)
        else:
            raise ValueError("Select GNN architecture between MG-GNN and Graph-Unet")
            
        directory  = test_args.model_dir

        model.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
        
        list_test = ['RAS', 'ML_ORAS']
        list_label = {'RAS':'RAS', 'ML_ORAS': 'MLORAS'}#
        dict_precs = {}
        dict_stats = {}
        n = grid.aggop[0].shape[0]
        
        x0 = np.random.random(grid.A.shape[0])
        x0 = x0/((grid.A@x0)**2).sum()**0.5
        
        data = grid.gdata
        data.edge_attr = data.edge_attr.float()
        model.eval()
        
        with torch.no_grad():
            out = model(data, grid, False)
            
        print('Passed to the network!')
        
        
        for name in list_test:

            M = preconditioner(grid, out, train = False, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
            if name == 'ML_ORAS':
                dict_precs[name] = [torch_2_scipy_sparse(M.detach()), out[1]]
            else:
                dict_precs[name] = [torch_2_scipy_sparse(M.detach()), None]
                
            dict_stats[name] = [M, out[1]]
    

        print('Obtained the preconditioners!')
        
        if test_args.precond:

            dict_loss = test_fgmres(grid, dict_precs, list_test)
            for name in list_test:
                
                plt.plot(dict_loss[name][:-2], label = list_label[name], marker='.')
         
    
            plt.xlabel("fGMRES Iteration")
            plt.ylabel("Residual norm")
            plt.yscale('log')
            plt.legend()
            plt.title('FGMRES '+str(int(grid.A.shape[0]))+'-node, '+str(grid.aggop[0].shape[-1])+' aggregates')
    
            plt.show()
            plt.figure()
        
        if test_args.stationary:

            dict_enorm, dict_vects = test_stats(grid, dict_stats, list_test)

            for name in list_test:
                
                plt.plot(dict_enorm[name], label = list_label[name], marker='.')
            
    
            plt.xlabel("Iteration")
            plt.ylabel("error norm")
            plt.yscale('log')
            plt.ylim([min(1e-5, dict_enorm['ML_ORAS'][-1]), 1])
            plt.title('Stationary '+str(int(grid.A.shape[0]))+'-node, '+str(grid.aggop[0].shape[-1])+' aggregates')
            plt.legend()
    
            plt.show()
            plt.figure()
        