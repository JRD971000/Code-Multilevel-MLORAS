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
num_lvl = 2


if num_lvl == 1:
    
    from utils_1L import *

if num_lvl == 2:
    
    from fgmres_2L import fgmres_2L
    from utils_2L import *
    
if num_lvl == 3:
    
    # from fgmres_2L import fgmres_2L
    from utils_3L import *

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
    
    u = torch.rand(grid.A.shape[0],1).double()
    u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
    K = 10
    
    
    dict_enorm = {}
    
    for name in list_test:
        
        dict_enorm[name] = []
        
        if name == 'Jacobi':
            MJ = torch.tensor(np.diag(grid.A.diagonal()**(-1)))
            dict_enorm[name] = test_stationary(grid, None, precond_type = None, u = u, K = K, M = MJ)
        
        else:
            dict_enorm[name] = test_stationary(grid, dict_stats[name], precond_type = name, u = u, K = K, M = dict_stats[name][0]) 
            
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
    
            lcmin = 0.0405#np.random.uniform(0.08, 0.09)
            lcmax = 0.0415#np.random.uniform(0.14, 0.15)
            n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
            randomized = True if np.random.rand() < 0.4 else True
            old_g = rand_grid_gen1(randomized = randomized, n = n, min_ = 0.03, min_sz = 0.6, 
                          lcmin = lcmin, lcmax = lcmax, distmin = 0.01, distmax = 0.035, PDE = test_args.PDE)
            
            print(old_g.A.shape[0])
            test_args.ratio = 25*((old_g.A.shape[0]/600)**0.5)/old_g.A.shape[0]
            grid =  Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops, 
                              cut=test_args.cut, h = 1, nu = 0, BC = test_args.BC)                
        
           
        # grid = torch.load('Data/data-800-1k/grid151.pth')
        # gg = Grid_PWA(grid.A, grid.mesh, ratio = 0.3, hops=-1)
        # gg.aggop_gen(grid.ratio, grid.cut, grid.aggop)
        # grid = torch.load('Data/TrainingGrids/grid9.pth')
        # grid = torch.load('grid_43k.pth')
        
        # A0 = grid.R0 @ grid.A @ grid.R0.transpose()
        # Cratio = max(2/A0.shape[0], 12*((A0.shape[0]/600)**0.5)/A0.shape[0])
        # Cgrid = Grid_PWA(A0, grid.mesh,Cratio)  
        # grid = torch.load('Data/testdata/grid9.pth')
        # grid.global_Lap_eig()
        
        print('Grid Made!')
    
        the_types = 1
            
        if the_types == 0:
    
            grid.aggop_gen(0.02, 1)
    
        if the_types == 2:
    
            grid.aggop_gen(35/grid.A.shape[0], 1)
        
        # grid = torch.load('data/Grids-Helmholtz-Dirichlet/grid90.pth')
        # gg = Grid_PWA(grid.A, grid.mesh, ratio = grid.ratio, hops=1)
        # gg.aggop_gen(grid.ratio, grid.cut, grid.aggop)
        # grid =  gg
        
        if test_args.plot:
    
            grid.plot_agg(size = 0.0, labeling = False, w = 0.1, shade=0.0008)
            # plt.savefig('../Paper-Multilevel-MLORAS/1_submitted_paper/figures/overTgrid.pdf', bbox_inches = 'tight')
    
            plt.show()
    
        ts = time.time()
    
        from NeuralNet import *
        from hgnn import HGNN
        model = HGNN(lvl=2, dim_embed=128, num_layers=2, K= 2, ratio=0.025, lr=1e-4)
        # model = mloras_net(dim = 128, K = 2, num_res = 8, num_convs = 4, lr = 1e-3, res = True, tf=False)
        
        # directory  = 'Models/Model-TG5k-new/model_epoch_best.pth'
        directory  = 'Models/Model-800-1k-neigh/model_epoch_best.pth'
    
        
        
        # from PrevNeuralNet import mloras_net
        # model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4)
    
        # directory  = 'Models/Model-Grids-Helmholtz-Dirichlet/trained_model.pth'
    
        # from NN_experiment import FC_test
        # model = FC_test(grid.A.shape[0], grid.gmask.nonzero()[0].shape[0], 128, lr = 1e-4)
    
        # directory  = 'Models/Model-gnn-all/model_epoch_best.pth'
        # directory  = 'Models/Model-NN-test/model_epoch989.pth'
    
    
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
        # =============================================================================
        #     
        # =============================================================================
        model_R = out[1].detach().to_dense().numpy()
    
        grid_R = grid.R0.toarray()
    
        plt.figure()
        start_col = 112
        rows = grid_R.shape[0]
        plt.imshow(grid_R[:rows, start_col:start_col+rows],cmap='seismic', 
                    interpolation='nearest', extent=[start_col,start_col+rows-1, 0,rows-1],
                    vmin = -abs(grid_R[:rows, start_col:start_col+rows]).max(), 
                    vmax = abs(grid_R[:rows, start_col:start_col+rows]).max())
    
        plt.colorbar()
        plt.show()
    
        plt.figure()
        plt.imshow(model_R[:rows, start_col:start_col+rows],cmap='seismic', 
                    interpolation='nearest', extent=[start_col,start_col+rows-1, 0,rows-1],
                    vmin = -abs(model_R[:rows, start_col:start_col+rows]).max(), 
                    vmax = abs(model_R[:rows, start_col:start_col+rows]).max())
    
        plt.colorbar()
        plt.show()
        # =============================================================================
        #     
        # =============================================================================
        print('Passed to the network!')
        for name in list_test:
            
            if num_lvl == 1:
                M = preconditioner(grid, out, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
                
                dict_precs[name] = M.numpy()
                dict_stats[name] = M
            
            if num_lvl == 2:
                M = preconditioner(grid, out, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
                if name == 'ML_ORAS':
                    dict_precs[name] = [M.to_dense().detach().numpy(), out[1]]
                else:
                    dict_precs[name] = [M.to_dense().detach().numpy(), None]
                    
                dict_stats[name] = [M, out[1]]
    
    
            if num_lvl == 3:
                
                grids = [grid, Cgrid]
                
                Cdata = Cgrid.gdata
                Cdata.edge_attr = Cdata.edge_attr.float()
                model.eval()
                with torch.no_grad():
                    Cout = model(Cdata, Cgrid)
                    
                    
                outs  = [out, Cout]
                
                M = preconditioner(grids, outs, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
                
                dict_precs[name] = M.numpy()
                dict_stats[name] = M
            
        print('Obtained the preconditioners!')
        if test_args.precond:
            t3 = time.time()
            dict_loss = test_fgmres(grid, dict_precs, list_test)
            t4 = time.time()
            print(f't43 = {t4-t3}\n')
            for name in list_test:
                
                plt.plot(dict_loss[name][:-2], label = list_label[name], marker='.')
         
    
            plt.xlabel("fGMRES Iteration")
            plt.ylabel("Residual norm")
            plt.yscale('log')
            plt.legend()
            plt.title('FGMRES '+str(int(grid.A.shape[0]))+'-node, '+str(grid.aggop[0].shape[-1])+' aggregates')
            # plt.savefig('../Paper-Multilevel-MLORAS/1_submitted_paper/figures/fgmres_interface.pdf', bbox_inches = 'tight')
    
            plt.show()
            plt.figure()
        
        if test_args.stationary:
            t5 = time.time()
            dict_enorm = test_stats(grid, dict_stats, list_test)
            t6 = time.time()
            print(f't65 = {t6-t5}\n')
            for name in list_test:
                
                plt.plot(dict_enorm[name], label = list_label[name], marker='.')
            
    
            tf = time.time()
            print('start-end = ', tf-ts)
            plt.xlabel("Iteration")
            plt.ylabel("error norm")
            plt.yscale('log')
            plt.ylim([min(1e-5, dict_enorm['ML_ORAS'][-1]), 1])
            plt.title('Sum != 1 Stationary '+str(int(grid.A.shape[0]))+'-node, '+str(grid.aggop[0].shape[-1])+' aggregates')
            plt.legend()
            # plt.savefig('../Paper-Multilevel-MLORAS/1_submitted_paper/figures/stationary_interface.pdf', bbox_inches = 'tight')
    
            plt.show()
            plt.figure()

 
def test_on_list():

    list_grids = []
    for iii in range(0,12):
        list_grids.append(torch.load('Data/testdata/grid'+str(iii)+'.pth'))
        
    for the_count in [8,9,10]:

        grid = list_grids[the_count]
        
        for the_types in range(1):
            # the_types = 1
            if the_types == 0:
                save_dir = 'Data/test_results/const'
                the_ratio = max(2/grid.A.shape[0], 0.02)
                grid.aggop_gen(the_ratio, 1)
                
            if the_types == 1:  
                save_dir = 'Data/test_results/square_root'
                the_ratio = max(12*((grid.A.shape[0]/600)**0.5)/grid.A.shape[0], 0.02)
                grid.aggop_gen(the_ratio, 1)
                
            if the_types == 2:
                save_dir = 'Data/test_results/linear'
                the_ratio = max(12/grid.A.shape[0], 0.02)
                grid.aggop_gen(the_ratio, 1)
                
            if test_args.plot:
        
                grid.plot_agg(size = 0.0, labeling = False, w = 0.1, shade=0.0008)
                
                plt.savefig('../Paper-Multilevel-MLORAS/Figs/grid'+str(the_count)+'.pdf', bbox_inches = 'tight')
                plt.show()
                
            ts = time.time()
            # model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4, res = True)
        
            # directory  = 'Models/'+test_args.model_dir+'/model_epoch'+str(test_args.epoch_num)+'.pth'
            # model.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
            
            list_test = ['RAS', 'ML_ORAS', 'ML_ORAS_Trans']
            list_label = {'RAS':'RAS','ML_ORAS': 'MLORAS', 'ML_ORAS_Trans':'MLORAS Transformer'}
            
            dict_precs = {}
            dict_stats = {}
            n = grid.aggop[0].shape[0]
            
            x0 = np.random.random(grid.A.shape[0])
            x0 = x0/((grid.A@x0)**2).sum()**0.5
            
            
            
            for name in list_test:
                
 
                    
                if name  == 'ML_ORAS' or name == 'RAS':
                    
                    from PrevNeuralNet import mloras_net
                    model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4)
                    
                    directory  = 'Models/Model-Grids-Helmholtz-Dirichlet/trained_model.pth'
                    
                    model.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
                    model.eval()
                    
                    with torch.no_grad():
                        
                        data = grid.gdata
                        data.edge_attr = data.edge_attr.float()
                        out = model(data, grid)
                        
                        M = preconditioner(grid, out, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
                        dict_precs[name] = M.numpy()
                        dict_stats[name] = M
                    
                if name == 'ML_ORAS_Trans':
                    
                    
                    from NeuralNet import mloras_net
                    model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4, res = True, tf=True)
                    
                    
                    directory  = 'Models/Model-trans/model_epoch_best1.pth'
                    model.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
                    model.eval()
                    
                    with torch.no_grad():
                        # grid.global_Lap_eig()
                        data = grid.gdata
                        data.edge_attr = data.edge_attr.float()
                        out = model(data, grid)
                        
                        M = preconditioner(grid, out, precond_type='ML_ORAS', u = torch.tensor(x0).unsqueeze(1))
                        dict_precs[name] = M.numpy()
                        dict_stats[name] = M
                
                
            
            if test_args.precond:
                
                dict_loss = test_fgmres(grid, dict_precs, list_test)
                torch.save(dict_loss, save_dir+'/dict_loss'+str(the_count)+'.pth')
                for name in list_test:
                    
                    plt.plot(dict_loss[name][:-2], label = name, marker='.')
             
        
                plt.xlabel("fGMRES Iteration")
                plt.ylabel("Residual norm")
                plt.yscale('log')
                plt.legend()
                plt.title('FGMRES convergence for '+str(int(grid.A.shape[0]))+'-node unstructured grid')
                
                plt.savefig('../Paper-Multilevel-MLORAS/Figs/fgmres'+str(the_count)+'.pdf', bbox_inches = 'tight')
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
        
                plt.savefig('../Paper-Multilevel-MLORAS/Figs/stationary'+str(the_count)+'.pdf', bbox_inches = 'tight')
                plt.show()
            
            

# test_on_list()