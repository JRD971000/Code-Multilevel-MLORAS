#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:00:46 2022

@author: alitaghibakhshi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:54:30 2022

@author: alitaghibakhshi
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
import os
from grids import *
#import fem
import sys
import torch as T
import copy
import random
from NeuralNet import *
from torch.utils.tensorboard import SummaryWriter
import scipy
from grids import *
import time
mpl.rcParams['figure.dpi'] = 300
from ST_CYR import *
import numpy as np
import scipy as sp
from pyamg import amg_core

def match_sparsiy(output, grid):
    
    sz = grid.gdata.x.shape[0]
    # out = torch.sparse_coo_tensor(grid.gdata.edge_index.tolist(), output.flatten(),(sz, sz)).to_dense()
    edges = [np.array(grid.mask_edges)[:,0].tolist(), np.array(grid.mask_edges)[:,1].tolist()]
    out = torch.sparse_coo_tensor(edges, output.flatten(),(sz, sz)).to_dense().double()
    # mask = torch.tensor(grid.gmask.toarray())
    
    return out #* mask

def get_Li (masked, grid):
    
    L_i = {}
    L = masked

    for i in range(grid.aggop[0].shape[-1]):

        nz = grid.list_cut_aggs[i]

        learnables = grid.learn_nodes[i]

        
        L_i[i] = torch.zeros(len(nz),len(nz)).double()

        list_idx = []

        for l in learnables:
            list_idx.append(nz.index(l))
            
        L_i[i][np.ix_(list_idx, list_idx)] = L[np.ix_(learnables, learnables)]
        

    return L_i


softmax = torch.nn.Softmax(dim=0)


def preconditioner(grid, output, train = False, precond_type = False, u = None, res = True):

    M = 0
        
    if precond_type == 'RAS':
                
        for i in range(grid.aggop[0].shape[-1]):
            
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            A_inv = torch.linalg.pinv(torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray()))

                
            modified_R_i = grid.modified_R[i].toarray()
            ####
            M += torch.tensor(modified_R_i.transpose()) @ A_inv @ torch.tensor(grid.R_hop[i].toarray())
        
    elif precond_type == 'ML_ORAS':
        
        # masked = match_sparsiy(output, grid)
        masked = output
        L = get_Li (masked, grid)
                
        for i in range(grid.aggop[0].shape[-1]):

            
            modified_R_i = grid.modified_R[i].toarray()
                
            modified_L = L[i]

            AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
            A_tilde_inv = torch.linalg.inv(AA + (1/(grid.h**2))*modified_L)
            M += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())


    else:
        raise RuntimeError('Wrong type for preconditioner: '+str(precond_type))
        
    return M

def stationary(grid, output, u = None, K = None, precond_type = 'ORAS'):

    M = preconditioner(grid, output, train = True, precond_type = precond_type, u = u)
    
    eprop = torch.eye(M.shape[0]) - M @ torch.tensor(grid.A.toarray())
    # return torch.norm(eprop)
    
    list_l2 = []
    out_lmax = copy.deepcopy(u)
    for k in range(K):
        out_lmax = eprop @ out_lmax
        l2 = torch.norm(out_lmax, p='fro', dim = 0)
        list_l2.append(l2)
    
    conv_fact = list_l2[-1]#(list_l2[-1]/list_l2[-3])**0.5
    L_max = torch.dot(softmax(conv_fact), conv_fact)

    return L_max

def Frob_loss(grid, output, u = None, K = None, precond_type = 'ORAS'):
    
    M = preconditioner(grid, output, train = True, precond_type = precond_type, u = u)

    eprop = torch.eye(M.shape[0]) - M @ torch.tensor(grid.A.toarray())
    return torch.norm(eprop)
    

def stationary_max(grid, output, u = None, K = None, precond_type = 'ORAS', res = True):

    M = preconditioner(grid, output, train = True, precond_type = precond_type, u = u)

    eprop = torch.eye(M.shape[0]) - M @ torch.tensor(grid.A.toarray())
    # return torch.norm(eprop)
    
    list_l2 = []
    out_lmax = copy.deepcopy(u)
    for k in range(K):
        out_lmax = eprop @ out_lmax
        l2 = torch.norm(out_lmax, p='fro', dim = 0)
        list_l2.append(l2)
    
    conv_fact = list_l2[-1]#(list_l2[-1]/list_l2[-3])**0.5
    L_max = max(conv_fact)#torch.dot(softmax(conv_fact), conv_fact)

    return L_max
        
def test_stationary(grid, output, precond_type, u, K, M=None,res = True):
    
    if M is None:
        M = preconditioner(grid, output, train = False, precond_type = precond_type,  u = u)
       
    eprop_a = torch.eye(M.shape[0]) - M @ torch.tensor(grid.A.toarray())
    
    out = copy.deepcopy(u)
    l2_list = []
    l2 = torch.norm( out, p='fro', dim = 0)
    l2_first = l2
    l2_list.append(l2.max())#torch.dot(softmax(l2), l2))
    for k in range(K):
        out = eprop_a @ out
        l2 = torch.norm(out, p='fro', dim = 0)
        l2_list.append(l2.max())#torch.dot(softmax(l2), l2))
        
    return l2_list

def struct_agg_PWA(n_row, n_col, agg_row, agg_col):


    arg0 = 0
    arg2 = []
    d = int(n_col/agg_col)
    
    for i in range(n_row * n_col):
        
        j = i%n_col
        k = i//n_col
        arg2.append(int(j//agg_col) + (k//agg_row)*d)
        
        
    arg0 = scipy.sparse.csr_matrix((np.ones(n_row * n_col), ([i for i in range(n_row * n_col)], arg2)), 
                                    shape=(n_row * n_col, max(arg2)+1))
            
    arg1 = np.zeros(max(arg2)+1)
    
    return (arg0, arg1, np.array(arg2))


