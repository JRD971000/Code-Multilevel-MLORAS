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
test_parser.add_argument('--BC', type=str, default='Dirichlet', help='Boundary conditions')
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
    
    from fgmres_2L import *
    from utils_2L_cpu import *
    from lloyd_gunet import *
    
if num_lvl == 3:
    
    # from fgmres_2L import fgmres_2L
    from utils_3L import *

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
    
    u = torch.rand(grid.A.shape[0],25).double()
    u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
    K = 20
    
    
    dict_enorm = {}
    dict_vects = {}
    for name in list_test:
        
        dict_enorm[name] = []
        dict_vects[name] = []
        
        if name == 'Jacobi':
            MJ = torch.tensor(np.diag(grid.A.diagonal()**(-1)))
            dict_enorm[name], dict_vects[name] = test_stationary(grid, None, precond_type = None, u = u, K = K, M = MJ)
        
        else:
            dict_enorm[name], dict_vects[name] = test_stationary(grid, dict_stats[name], precond_type = name, u = u, K = K, M = dict_stats[name][0]) 
            
    return dict_enorm, dict_vects


if __name__ =='__main__':
    
        # stat_results = {}
        # gmres_results ={}
        # gmres_results['ML_ORAS'] = {}
        # gmres_results['RAS'] = {}
        for icount in range(1):
            # icount = 10
            # grid = torch.load('Data/test_grids/grid'+str(icount)+'.pth')
            grid = torch.load('grid1.pth')
            # g1 = Grid_PWA(grid.A, grid.mesh, 0.1, hops = 1, cut = 2)
        
            # g1.aggop_gen(0.1, 2, grid.aggop)
            # grid = g1
            # grid = refine_grid(grid, 2)
            

            # if test_args.structured:
                
            #     ratio = test_args.ratio
            #     n = 30#test_args.size_structured
                
            #     if test_args.BC == 'Dirichlet':
            #         old_g = structured(n, n, Neumann=False)
            #     else:
            #         old_g = structured(n, n, Neumann=True)  
                    
            #     grid =  Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops, 
            #                       cut=test_args.cut, h = 1/(n+1), nu = 1)#, BC = 'Dirichlet')
            #     grid.aggop_gen(ratio = 0.1, cut = 1, node_agg = struct_agg_PWA(n,n,10,10))
                
            # else:
        
            #     lcmin = 0.0535#np.random.uniform(0.08, 0.09)
            #     lcmax = 0.0541#np.random.uniform(0.14, 0.15)
            #     n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
            #     randomized = True #if np.random.rand() < 0.4 else True
            #     old_g = rand_grid_gen1(randomized = randomized, n = n, min_ = 0.03, min_sz = 0.6, 
            #                   lcmin = lcmin, lcmax = lcmax, distmin = 0.01, distmax = 0.035, PDE = test_args.PDE)
                
            #     print(old_g.A.shape[0])
            #     test_args.ratio = 25*((old_g.A.shape[0]/600)**0.5)/old_g.A.shape[0]
            #     grid =  Grid_PWA(old_g.A, old_g.mesh, test_args.ratio, hops = test_args.hops, 
            #                       cut=test_args.cut, h = 1, nu = 0, BC = test_args.BC)                
            
               
            # grid = torch.load('Data/data-800-1k/grid0.pth')
            # gg = Grid_PWA(grid.A, grid.mesh, ratio = 0.3, hops=-1)
            # gg.aggop_gen(grid.ratio, grid.cut, grid.aggop)
            # grid = torch.load('Data/TrainingGrids/grid9.pth')
            # grid = torch.load('grid.pth')
            
            # A0 = grid.R0 @ grid.A @ grid.R0.transpose()
            # Cratio = max(2/A0.shape[0], 12*((A0.shape[0]/600)**0.5)/A0.shape[0])
            # Cgrid = Grid_PWA(A0, grid.mesh,Cratio)  
            # grid = torch.load('Data/testdata/grid9.pth')
            # grid.global_Lap_eig()
            
            print('Grid Made!')
            print(grid.A.shape)
            the_types = 1
                
            if the_types == 0:
        
                grid.aggop_gen(0.02, 1)
        
            if the_types == 2:
        
                grid.aggop_gen(35/grid.A.shape[0], 1)
            
            # grid = torch.load('data/Grids-Helmholtz-Dirichlet/grid90.pth')
            # gg = Grid_PWA(grid.A, grid.mesh, ratio = grid.ratio, hops=1)
            # gg.aggop_gen(grid.ratio, grid.cut, grid.aggop)
            # grid =  gg
            
            # if test_args.plot:
        
            #     grid.plot_agg(size = 0.0, labeling = False, w = 0.1, shade=0.0008)
            #     # plt.savefig('../Paper-Multilevel-MLORAS/1_submitted_paper/figures/overTgrid.pdf', bbox_inches = 'tight')
        
            #     plt.show()
        
            ts = time.time()
        
            from NeuralNet import *
            from hgnn import HGNN
            model = HGNN(lvl=2, dim_embed=128, num_layers=4, K= 2, ratio=0.2, lr=1e-4)

            directory = 'model_epoch_best.pth'
            
            
            # from PrevNeuralNet import mloras_net
            # model = mloras_net(dim = test_args.dim, K = test_args.TAGConv_k, num_res = 8, num_convs = 4, lr = 1e-4)
        
            # directory  = 'Models/Model-Grids-Helmholtz-Dirichlet/trained_model.pth'
        
            # from NN_experiment import FC_test
            # model = FC_test(grid.A.shape[0], grid.gmask.nonzero()[0].shape[0], 128, lr = 1e-4)
        
            # directory  = 'Models/Model-gnn-all/model_epoch_best.pth'
            # directory  = 'Models/Model-NN-test/model_epoch989.pth'
        
        
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
                
            t2 = time.time()
            print('Passed to the network!, Time = ', t2 - ts)
            
            # =============================================================================
            #     
            # =============================================================================
            # model_R = out[1].detach().to_dense().numpy()
        
            # grid_R = grid.R0.toarray()
        
            # plt.figure()
            # start_col = 112
            # rows = grid_R.shape[0]
            # plt.imshow(grid_R[:rows, start_col:start_col+rows],cmap='seismic', 
            #             interpolation='nearest', extent=[start_col,start_col+rows-1, 0,rows-1],
            #             vmin = -abs(grid_R[:rows, start_col:start_col+rows]).max(), 
            #             vmax = abs(grid_R[:rows, start_col:start_col+rows]).max())
        
            # plt.colorbar()
            # plt.show()
        
            # plt.figure()
            # plt.imshow(model_R[:rows, start_col:start_col+rows],cmap='seismic', 
            #             interpolation='nearest', extent=[start_col,start_col+rows-1, 0,rows-1],
            #             vmin = -abs(model_R[:rows, start_col:start_col+rows]).max(), 
            #             vmax = abs(model_R[:rows, start_col:start_col+rows]).max())
        
            # plt.colorbar()
            # plt.show()
            # =============================================================================
            #     
            # =============================================================================
            
            for name in list_test:
                
                if num_lvl == 1:
                    M = preconditioner(grid, out, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
                    
                    dict_precs[name] = M.numpy()
                    dict_stats[name] = M
                
                if num_lvl == 2:
                    M = preconditioner(grid, out, train = False, precond_type=name, u = torch.tensor(x0).unsqueeze(1))
                    if name == 'ML_ORAS':
                        dict_precs[name] = [torch_2_scipy_sparse(M.detach()), out[1]]
                    else:
                        dict_precs[name] = [torch_2_scipy_sparse(M.detach()), None]
                        
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
                
            t3 = time.time()
            print('Obtained the preconditioners!, Time = ', t3-t2)
            
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
                dict_enorm, dict_vects = test_stats(grid, dict_stats, list_test)
                t6 = time.time()
                print(f'Stationary Test Time = {t6-t5}\n')
                for name in list_test:
                    
                    plt.plot(dict_enorm[name], label = list_label[name], marker='.')
                
        
                tf = time.time()
                print('start-end = ', tf-ts)
                plt.xlabel("Iteration")
                plt.ylabel("error norm")
                plt.yscale('log')
                plt.ylim([min(1e-5, dict_enorm['ML_ORAS'][-1]), 1])
                plt.title('Stationary '+str(int(grid.A.shape[0]))+'-node, '+str(grid.aggop[0].shape[-1])+' aggregates')
                plt.legend()
                # plt.savefig('../Paper-Multilevel-MLORAS/1_submitted_paper/figures/stationary_interface.pdf', bbox_inches = 'tight')
        
                plt.show()
                plt.figure()
            
        #     stat_results[icount] = dict_enorm
        #     gmres_results['ML_ORAS'][icount] = len(dict_loss['ML_ORAS'])
        #     gmres_results['RAS'][icount] = len(dict_loss['RAS'])

        # torch.save(stat_results, 'result_lg_lvl2_layer1_copy.pth')
        # torch.save(gmres_results, 'gmres_lg_lvl2_layer1_copy.pth')

# plot_color(grid, 10, 0.0, False, 1, dict_vects['RAS'][200].numpy())
# plot_color(grid, 10, 0.0, False, 1, dict_vects['ML_ORAS'][200].numpy())

# u = torch.tensor(np.random.rand(grid.A.shape[0]))
# plot_color(grid, 10, 0.0, False, 1, u.numpy())
# plt.title('initial')
# plt.show()
# for i in range(120):
#     plt.figure()
#     u = stat_cyc(make_sparse_torch(grid.A).to_sparse_csr(), dict_stats['ML_ORAS'][0], out[1].to_sparse_csr(), u, False, 'fine')
#     plot_color(grid, 10, 0.0, False, 1, u.numpy())
#     plt.title('MLORAS fine'+str(i))
#     plt.show()
    
#     u = stat_cyc(make_sparse_torch(grid.A).to_sparse_csr(), dict_stats['ML_ORAS'][0], out[1].to_sparse_csr(), u, False, 'coarse')
#     plt.figure()
#     plot_color(grid, 10, 0.0, False, 1, u.numpy())
#     plt.title('MLORAS coarse'+str(i))
#     plt.show()