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
    out = torch.sparse_coo_tensor(grid.gdata.edge_index.tolist(), output.flatten(),(sz, sz)).to_dense()
    mask = torch.tensor(grid.gmask.toarray())
    
    return out * mask

def get_Li (masked, grid):
    
    L_i = {}
    L = masked

    for i in range(grid.aggop[0].shape[-1]):
        
        nz = grid.R_hop[i].nonzero()[-1].tolist()
        learnables = grid.learn_nodes[i]
        
        Local_mask = torch.zeros_like(L)
        Local_mask[np.ix_(learnables, learnables)] = 1.0
        Lmask = Local_mask * L
        
        L_i[i] = Lmask[np.ix_(nz, nz)]
        
    return L_i

def get_Bij(L_i, grid):
    
    B = {}
    n = grid.aggop[0].shape[-1]
    for i in range(n):
        for j in range(n):
            if i!=j:
                B[i,j] = -torch.tensor((grid.R[i] @ grid.R[j].transpose() @ grid.A_i[j]).toarray()) + \
                    grid.h * L_i[i] @ torch.tensor((grid.R[i] @ grid.R_tilde[j].transpose()).toarray())
    return B



softmax = torch.nn.Softmax(dim=0)


def preconditioner(grid, model, train = False, precond_type = False, u = None, res = True):
    
    
    if precond_type == 'Brute_Force':

        M = 0
        data = model.forward()
        n = int((data.shape[0] + 1)/2)
        main_d_0 = torch.diag(data[:n], 0)
        sub_d_0 = torch.diag(data[n:], -1)
        sup_d_0 = torch.diag(data[n:], +1)
        l0 = main_d_0 + sub_d_0 + sup_d_0
        
        main_data = data[:n]
        main_idx =[n-1-i for i in range(n)]
        main_data = main_data[main_idx]
        subp_data = data[n:]
        subp_idx =[n-2-i for i in range(n-1)]
        subp_data = subp_data[subp_idx]
        
        main_d_1 = torch.diag(main_data, 0)
        sub_d_1 = torch.diag(subp_data, -1)
        sup_d_1 = torch.diag(subp_data, +1)
        l1 = main_d_1 + sub_d_1 + sup_d_1
        
        L = {}
        L[0] = torch.zeros((n*(1+int(n/2)), n*(1+int(n/2))))
        L[1] = torch.zeros((n*(1+int(n/2)), n*(1+int(n/2))))
        L[0][-n:, -n:] = l0
        L[1][:n, :n] = l1
        
        for i in range(grid.aggop[0].shape[-1]):

            r0 = grid.R[i].toarray().nonzero()[-1].tolist()
            rdelta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            list_ixs = []
            for e in r0:
                list_ixs.append(rdelta.index(e))
                
            modified_R_i = np.zeros_like(torch.tensor(grid.R_hop[i].toarray()))
            modified_R_i[list_ixs, :] = grid.R[i].toarray()

            nz_0 = grid.R[i].toarray().nonzero()[-1].tolist()
            nz_delta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            nonz = []
            for elem in nz_0:
                nonz.append(nz_delta.index(elem))
                
            modified_L = L[i]

            AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
            A_tilde_inv = torch.linalg.inv(AA + (1/(grid.h**2))*modified_L)
            M += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())
        
        return M
            
    if train:
        
        data = grid.gdata
        # data.x = data.x.float()
        # data.x = torch.cat(((grid.data().x, torch.tensor(u))), dim =1).float()
        data.edge_attr = data.edge_attr.float()
        model.float()
        output = model.forward(data, grid)
    else:
        
        with torch.no_grad():
            data = grid.gdata
            # data.x = data.x.float()
            # data.x = torch.cat(((grid.data().x, torch.tensor(u))), dim =1).float()
            data.edge_attr = data.edge_attr.float()
            model.float()
            output = model.forward(data, grid)


    masked = match_sparsiy(output, grid)

    L = get_Li (masked, grid)

    M = 0

    if precond_type == 'AS':  
        for i in range(grid.aggop[0].shape[-1]):
        
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()))
            A_inv = torch.linalg.pinv(torch.tensor(grid.R[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R[i].transpose().toarray()))
            M += torch.tensor(grid.R[i].transpose().toarray()) @ A_inv @ torch.tensor(grid.R[i].toarray())
        M = torch.tensor(M)
        
        
    elif precond_type == 'RAS':
        for i in range(grid.aggop[0].shape[-1]):
            
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            A_inv = torch.linalg.pinv(torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray()))
  
            r0 = grid.R[i].toarray().nonzero()[-1].tolist()
            rdelta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            list_ixs = []
            for e in r0:
                list_ixs.append(rdelta.index(e))
                
            modified_R_i = np.zeros_like(torch.tensor(grid.R_hop[i].toarray()))
            modified_R_i[list_ixs, :] = grid.R[i].toarray()
            ####
            M += torch.tensor(modified_R_i.transpose()) @ A_inv @ torch.tensor(grid.R_hop[i].toarray())
        
    elif precond_type == 'ORAS': 
        for i in range(grid.aggop[0].shape[-1]):
            
            ####
            nz_0 = grid.R[i].toarray().nonzero()[-1].tolist()
            nz_delta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            nonz = []
            for elem in nz_0:
                nonz.append(nz_delta.index(elem))
                
            modified_L = torch.zeros(len(nz_delta), len(nz_delta)).double()
            modified_L[np.ix_(nonz, nonz)] = L[i]
        
            
            r0 = grid.R[i].toarray().nonzero()[-1].tolist()
            rdelta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            list_ixs = []
            for e in r0:
                list_ixs.append(rdelta.index(e))
                
            modified_R_i = np.zeros_like(torch.tensor(grid.R_hop[i].toarray()))
            modified_R_i[list_ixs, :] = grid.R[i].toarray()
 
            A_tilde_inv = torch.linalg.pinv((torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())) + (1/(grid.h**2))*modified_L)
            
            M += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())

    elif precond_type == 'ML_ORAS':

        for i in range(grid.aggop[0].shape[-1]):

            r0 = grid.R[i].toarray().nonzero()[-1].tolist()
            rdelta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            list_ixs = []
            for e in r0:
                list_ixs.append(rdelta.index(e))
                
            modified_R_i = np.zeros_like(torch.tensor(grid.R_hop[i].toarray()))
            modified_R_i[list_ixs, :] = grid.R[i].toarray()

            nz_0 = grid.R[i].toarray().nonzero()[-1].tolist()
            nz_delta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            nonz = []
            for elem in nz_0:
                nonz.append(nz_delta.index(elem))
                
            modified_L = L[i]

            AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
            A_tilde_inv = torch.linalg.inv(AA + (1/(grid.h**2))*modified_L)
            M += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())

    elif precond_type == 'ORAS_OO2':
                
        overlap = get_overlaps(grid)
        size = len(overlap[0])
        
        ## OO0
        # p  = (2**(-1/3)) * ((np.pi**2 + grid.nu) ** (1/3)) * (grid.h ** (-1/3))
        # q = 0
        
        ##OO2
        p  = (2**(-3/5)) * ((np.pi**2 + grid.nu) ** (2/5)) * (grid.h ** (-1/5))
        q = (2**(-1/5)) * ((np.pi**2 + grid.nu) ** (-1/5)) * (grid.h ** (3/5))

        I = torch.eye(size)
        T0 = torch.tensor(scipy.sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(size, size)).toarray())
        Tn = T0 + grid.nu*(grid.h ** 2)*I
        T_tilde = 0.5*Tn + p*grid.h*I + q*(T0-2*I)/grid.h
        
        for i in range(grid.aggop[0].shape[-1]):
            
            list_domain = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            list_idx = []
            for e in overlap[i]:
                if e in list_domain:
                    list_idx.append(list_domain.index(e))
                
            r0 = grid.R[i].toarray().nonzero()[-1].tolist()
            rdelta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            list_ixs = []
            for e in r0:
                list_ixs.append(rdelta.index(e))
                
            modified_R_i = np.zeros_like(torch.tensor(grid.R_hop[i].toarray()))
            modified_R_i[list_ixs, :] = grid.R[i].toarray()

            nz_0 = grid.R[i].toarray().nonzero()[-1].tolist()
            nz_delta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            nonz = [] 
            for elem in nz_0:
                nonz.append(nz_delta.index(elem))
                         
            ####
            
            # A_tilde_inv = torch.zeros(len(nz_delta), len(nz_delta)k).double()
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            # A_tilde_inv[np.ix_(nonz, nonz)] = A_inv
            
            
            AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
            AA[np.ix_(list_idx, list_idx)] = T_tilde/(grid.h ** 2)
            # A_tilde_inv = torch.linalg.pinv(AA)
            A_tilde_inv = torch.linalg.pinv(AA)
            
            M += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())
            
    elif precond_type == 'ORAS_OO0':
                
        overlap = get_overlaps(grid)
        size = len(overlap[0])
        
        ## OO0
        p  = (2**(-1/3)) * ((np.pi**2 + grid.nu) ** (1/3)) * (grid.h ** (-1/3))
        q = 0
        
        ##OO2
        # p  = (2**(-3/5)) * ((np.pi**2 + grid.nu) ** (2/5)) * (grid.h ** (-1/5))
        # q = (2**(-1/5)) * ((np.pi**2 + grid.nu) ** (-1/5)) * (grid.h ** (3/5))

        I = torch.eye(size)
        T0 = torch.tensor(scipy.sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(size, size)).toarray())
        Tn = T0 + grid.nu*(grid.h ** 2)*I
        T_tilde = 0.5*Tn + p*grid.h*I + q*(T0-2*I)/grid.h
        
        for i in range(grid.aggop[0].shape[-1]):
            
            list_domain = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            list_idx = []
            for e in overlap[i]:
                if e in list_domain:
                    list_idx.append(list_domain.index(e))
                
            r0 = grid.R[i].toarray().nonzero()[-1].tolist()
            rdelta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            list_ixs = []
            for e in r0:
                list_ixs.append(rdelta.index(e))
                
            modified_R_i = np.zeros_like(torch.tensor(grid.R_hop[i].toarray()))
            modified_R_i[list_ixs, :] = grid.R[i].toarray()

            nz_0 = grid.R[i].toarray().nonzero()[-1].tolist()
            nz_delta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
            nonz = [] 
            for elem in nz_0:
                nonz.append(nz_delta.index(elem))
                         
            ####
            
            # A_tilde_inv = torch.zeros(len(nz_delta), len(nz_delta)k).double()
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            # A_tilde_inv[np.ix_(nonz, nonz)] = A_inv
            
            
            AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
            AA[np.ix_(list_idx, list_idx)] = T_tilde/(grid.h ** 2)
            # A_tilde_inv = torch.linalg.pinv(AA)
            A_tilde_inv = torch.linalg.pinv(AA)
            M += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())
            

    else:
        raise RuntimeError('Wrong type for preconditioner: '+str(precond_type))
        
    return M

def stationary(grid, model, u = None, K = None, precond_type = 'ORAS'):

    M = preconditioner(grid, model, train = True, precond_type = precond_type, u = u)
    
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

def Frob_loss(grid, model, u = None, K = None, precond_type = 'ORAS'):
    
    M = preconditioner(grid, model, train = True, precond_type = precond_type, u = u)

    eprop = torch.eye(M.shape[0]) - M @ torch.tensor(grid.A.toarray())
    return torch.norm(eprop)
    

def stationary_max(grid, model, u = None, K = None, precond_type = 'ORAS', res = True):

    M = preconditioner(grid, model, train = True, precond_type = precond_type, u = u)

    eprop = torch.eye(M.shape[0]) - M @ torch.tensor(grid.A.toarray())
    return torch.norm(eprop)
    
    list_l2 = []
    out_lmax = copy.deepcopy(u)
    for k in range(K):
        out_lmax = eprop @ out_lmax
        l2 = torch.norm(out_lmax, p='fro', dim = 0)
        list_l2.append(l2)
    
    conv_fact = list_l2[-1]#(list_l2[-1]/list_l2[-3])**0.5
    L_max = max(conv_fact)#torch.dot(softmax(conv_fact), conv_fact)

    return L_max
        
def test_stationary(grid, model, precond_type, u, K, M=None,res = True):
    
    if M is None:
        M = preconditioner(grid, model, train = False, precond_type = precond_type,  u = u)
       
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


