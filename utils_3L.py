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


def preconditioner(grids, outputs, train = False, precond_type = False, u = None, res = True):

    grid, Cgrid = grids
    output, Coutput = outputs
        
    if precond_type == 'RAS':
        M1_ORAS = 0
        
        for i in range(grid.aggop[0].shape[-1]):
            
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            A_inv = torch.linalg.pinv(torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray()))

                
            modified_R_i = grid.modified_R[i].toarray()
            ####
            M1_ORAS += torch.tensor(modified_R_i.transpose()) @ A_inv @ torch.tensor(grid.R_hop[i].toarray())
        
        D0 = 0
        for i in range(Cgrid.aggop[0].shape[-1]):
            
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            C_A_inv = torch.linalg.pinv(torch.tensor(Cgrid.R_hop[i].toarray()) @ torch.tensor(Cgrid.A.toarray()) @ torch.tensor(Cgrid.R_hop[i].transpose().toarray()))

                
            C_modified_R_i = Cgrid.modified_R[i].toarray()
            ####
            D0 += torch.tensor(C_modified_R_i.transpose()) @ C_A_inv @ torch.tensor(Cgrid.R_hop[i].toarray())
            
        t_R0 = torch.tensor(grid.R0.toarray())
        t_A  = torch.tensor(grid.A.toarray())
        R00 = Cgrid.R0
        A0  = Cgrid.A
        A0_inv = torch.linalg.pinv(torch.tensor(A0.toarray()))
        A00 = R00 @ A0 @ R00.transpose()
        A00_inv  = scipy.sparse.linalg.inv(A00)
        I0  = torch.eye(D0.shape[0])
        M0  = (I0 - torch.tensor((R00.transpose() @ A00_inv @ R00 @ A0).toarray())) @ (I0 - D0 @ torch.tensor(A0.toarray()))
        
        I = torch.eye(M1_ORAS.shape[0]) 
        M3_ORAS = (I - t_R0.t() @ (I0-M0) @ A0_inv @ t_R0 @ t_A) @ (I - M1_ORAS @ t_A)
        
        return M3_ORAS
        
    elif precond_type == 'ML_ORAS':
        M1_ORAS = 0
        # masked = match_sparsiy(output, grid)
        masked = output
        L = get_Li (masked, grid)
                
        for i in range(grid.aggop[0].shape[-1]):

            
            modified_R_i = grid.modified_R[i].toarray()
                
            modified_L = L[i]

            AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
            A_tilde_inv = torch.linalg.pinv(AA + (1/(grid.h**2))*modified_L)
            M1_ORAS += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())
            
        # D0 = 0
        
        # Cmasked = Coutput
        # C_L = get_Li (Cmasked, Cgrid)
        
        # t_C_A = torch.tensor(Cgrid.A.toarray())
        
        # for i in range(Cgrid.aggop[0].shape[-1]):

            
        #     C_modified_R_i = Cgrid.modified_R[i].toarray()
                
        #     C_modified_L = C_L[i]

        #     C_AA = torch.tensor(Cgrid.R_hop[i].toarray()) @ t_C_A @ torch.tensor(Cgrid.R_hop[i].transpose().toarray())
        #     C_A_tilde_inv = torch.linalg.pinv(C_AA + (1/(Cgrid.h**2))*C_modified_L)
        #     D0 += torch.tensor(C_modified_R_i.transpose()) @ C_A_tilde_inv @ torch.tensor(Cgrid.R_hop[i].toarray())
        
        D0 = 0
        for i in range(Cgrid.aggop[0].shape[-1]):
            
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            C_A_inv = torch.linalg.pinv(torch.tensor(Cgrid.R_hop[i].toarray()) @ torch.tensor(Cgrid.A.toarray()) @ torch.tensor(Cgrid.R_hop[i].transpose().toarray()))

                
            C_modified_R_i = Cgrid.modified_R[i].toarray()
            ####
            D0 += torch.tensor(C_modified_R_i.transpose()) @ C_A_inv @ torch.tensor(Cgrid.R_hop[i].toarray())
            
        
        t_R0 = torch.tensor(grid.R0.toarray())
        t_A  = torch.tensor(grid.A.toarray())
        R00 = Cgrid.R0
        A0  = Cgrid.A
        A0_inv = torch.linalg.pinv(torch.tensor(A0.toarray()))
        A00 = R00 @ A0 @ R00.transpose()
        A00_inv  = scipy.sparse.linalg.inv(A00)
        I0  = torch.eye(D0.shape[0])
        M0  = (I0 - torch.tensor((R00.transpose() @ A00_inv @ R00 @ A0).toarray())) @ (I0 - D0 @ torch.tensor(A0.toarray()))
        
        I = torch.eye(M1_ORAS.shape[0]) 
        M3_ORAS = (I - t_R0.t() @ (I0-M0) @ A0_inv @ t_R0 @ t_A) @ (I - M1_ORAS @ t_A)
        
        return M3_ORAS
    
    else:
        raise RuntimeError('Wrong type for preconditioner: '+str(precond_type))
        return

def stationary(grid, output, u = None, K = None, precond_type = 'ORAS'):

    eprop = preconditioner(grid, output, train = True, precond_type = precond_type, u = u)
    
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
    
    
    M_RAS = preconditioner(grid, output, train = True, precond_type = 'RAS', u = u)

    eprop_RAS = torch.eye(M_RAS.shape[0]) - M_RAS @ torch.tensor(grid.A.toarray())
    # return torch.norm(eprop)
    
    list_l2_RAS = []
    out_lmax_RAS = copy.deepcopy(u)
    for k in range(K):
        out_lmax_RAS = eprop_RAS @ out_lmax_RAS
        l2_RAS = torch.norm(out_lmax_RAS, p='fro', dim = 0)
        list_l2_RAS.append(l2_RAS)
    
    conv_fact_RAS = list_l2_RAS[-1]#(list_l2[-1]/list_l2[-3])**0.5
    L_max_RAS = max(conv_fact_RAS)#torch.dot(softmax(conv_fact), conv_fact)
    
    L_max = L_max/L_max_RAS

    return L_max
        
def test_stationary(grid, output, precond_type, u, K, M=None,res = True):
    
    if M is None:
        M = preconditioner(grid, output, train = False, precond_type = precond_type,  u = u)
       
    
    out = copy.deepcopy(u)
    l2_list = []
    l2 = torch.norm( out, p='fro', dim = 0)

    l2_list.append(l2.max())#torch.dot(softmax(l2), l2))
    for k in range(K):
        out = M @ out
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


