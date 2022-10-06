#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:35:42 2022

@author: alitaghibakhshi
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear, ReLU
import os
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.pool import SAGPooling, TopKPooling, ASAPooling, MemPooling, PANPooling
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear,\
                                 GraphUNet, TAGConv, MessagePassing
import sys
import scipy
import torch_geometric
from torch_geometric.data import HeteroData
import torch.optim as optim
from pyamg.aggregation import lloyd_aggregation
from torch.nn.functional import relu, sigmoid

from NeuralNet import EdgeModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
def make_sparse_torch(A, sparse = True):
    if sparse:
        idxs = torch.tensor(np.array(A.nonzero()))
        dat = torch.tensor(A.data).float()
    else:
        idxs = torch.tensor([[i//A.shape[1] for i in range(A.shape[0]*A.shape[1])], 
                             [i% A.shape[1] for i in range(A.shape[0]*A.shape[1])]])
        dat = A.flatten()
    s = torch.sparse_coo_tensor(idxs, dat, (A.shape[0], A.shape[1]))
    return s.to_sparse_csr()


class PlainMP(MessagePassing):
    def __init__(self, aggr = 'mean'):
        super().__init__(aggr=aggr) #  "Max" aggregation.
 

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):

        return x_i

class GUNET(torch.nn.Module):
    def __init__(self, in_dim, dim, out_dim, depth, lr):
        super().__init__()
        self.gunet = GraphUNet(in_dim, dim, out_dim, depth, pool_ratios=[0.5 for i in range(depth)])
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr = lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.device = torch.device('cpu')
        self.to(self.device)
    
    def forward(self, data):
        
        return self.gunet(data.x, data.edge_index)

def K_means_agg(X, A, ratio, grid):
    
    # R, idx = lloyd_aggregation(A, ratio)

    R = grid.R0.transpose()
    
    # tsR = make_sparse_torch(R).to(device)#.float()
    # tsA = make_sparse_torch(A).to(device)#.float()
    idx = grid.aggop[1]
    
    idx = torch.tensor(idx).to(device)
    A_coarse = R.transpose() @ A @ R
    # A_coarse = tsR.t() @ tsA @ tsR
    coarse_index = torch.tensor(np.int64(A_coarse.nonzero())).to(device)
    # coarse_index = A_coarse.to_dense().nonzero().to(device)
    # coarse_index = coarse_index.reshape(coarse_index.shape[1], coarse_index.shape[0])
    
    X_coarse = (X.t() @ torch.tensor(R.toarray()).float()).t()
    # X_coarse = (X.t() @ tsR.to_dense()).t()

    
    fine2coarse = torch.tensor(np.int64(R.nonzero())).to(device)
    # fine2coarse = tsR.to_dense().nonzero().to(device)
    # fine2coarse = fine2coarse.reshape(fine2coarse.shape[1], fine2coarse.shape[0])
    
    attr_coarse = torch.tensor(A_coarse.toarray().flatten()[A_coarse.toarray().flatten().nonzero()]).unsqueeze(1).float().to(device)
    # attr_coarse = A_coarse.to_dense().flatten()[A_coarse.to_dense().flatten().nonzero()].float().to(device)

    return X_coarse, coarse_index, idx, fine2coarse, attr_coarse, A_coarse
    
def K_means_agg_torch(X, A, ratio, grid):
    
    # R, idx = lloyd_aggregation(A, ratio)

    R = grid.R0.transpose()
    
    tR = torch.tensor(R.toarray()).float().to(device)#.float()
    tA = torch.tensor(A.toarray()).float().to(device)#.float()
    idx = grid.aggop[1]
    
    idx = torch.tensor(idx).to(device)
    # A_coarse = R.transpose() @ A @ R
    A_coarse = tR.t() @ tA @ tR
    A_coarse = scipy.sparse.csr_matrix(np.array(A_coarse.cpu()))
    
    coarse_index = torch.tensor(np.int64(A_coarse.nonzero())).to(device)
    # coarse_index = A_coarse.to_dense().nonzero().to(device)
    # coarse_index = coarse_index.reshape(coarse_index.shape[1], coarse_index.shape[0])
    
    X_coarse = (X.t() @ torch.tensor(R.toarray()).float().to(device)).t()
    # X_coarse = (X.t() @ tsR.to_dense()).t()

    
    fine2coarse = torch.tensor(np.int64(R.nonzero())).to(device)
    # fine2coarse = tsR.to_dense().nonzero().to(device)
    # fine2coarse = fine2coarse.reshape(fine2coarse.shape[1], fine2coarse.shape[0])
    
    attr_coarse = torch.tensor(A_coarse.toarray().flatten()[A_coarse.toarray().flatten().nonzero()]).unsqueeze(1).float().to(device)
    # attr_coarse = A_coarse.to_dense().flatten()[A_coarse.to_dense().flatten().nonzero()].float().to(device)

    return X_coarse, coarse_index, idx, fine2coarse, attr_coarse, A_coarse


class HGNN(torch.nn.Module):
    def __init__(self, lvl, dim_embed, num_layers, K, ratio, lr):
        super().__init__()
        
        
        self.lvl = lvl

        # self.droupout = droupout
        self.ratio = ratio
        
        self.pre_edge_main = torch.nn.Sequential(Linear(2, dim_embed), ReLU(), 
                                    Linear(dim_embed, dim_embed), ReLU(),
                                    Linear(dim_embed, 1))
        
        self.pre_edge = torch.nn.Sequential(Linear(1, dim_embed), ReLU(), 
                                    Linear(dim_embed, dim_embed), ReLU(),
                                    Linear(dim_embed, 1))
        
        self.pre_node = torch.nn.Sequential(Linear(1, dim_embed), ReLU(), 
                                    Linear(dim_embed, dim_embed), ReLU(),
                                    Linear(dim_embed, dim_embed))
        
        
        self.convs = torch.nn.ModuleList()
        self.f2c   = torch.nn.ModuleList()
        self.c2f   = torch.nn.ModuleList()
        self.name  = 'hgnn_lvl'+str(lvl)+'_numlayer'+str(num_layers)
        normalizations = []

        
        for _ in range(num_layers):
            
            dict_fc = {}
            dict_cf = {}
            dict_ff = {}
            
            for i in range(lvl):
                if i <lvl-1:
                    dict_fc[('L'+str(i), '->', 'L'+str(i+1))] = PlainMP()#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)#TAGConv(dim_embed, dim_embed, K = 2, normalize = False)#
                    dict_cf[('L'+str(i+1), '->', 'L'+str(i))] = PlainMP()#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)#TAGConv(dim_embed, dim_embed, K = 2, normalize = False)#
                    
            conv_fc = HeteroConv(dict_fc, aggr='sum')
            conv_cf = HeteroConv(dict_cf, aggr='sum')
            
            self.f2c.append(conv_fc)            
            self.c2f.append(conv_cf)

            if self.lvl > 1:
                
                for i in range(lvl):
                    if i == 0 or i == lvl-1:
         
                        dict_ff[('L'+str(i), '-', 'L'+str(i))] = TAGConv(2*dim_embed, dim_embed, K = K, normalize = False)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)
                    else:
                        dict_ff[('L'+str(i), '-', 'L'+str(i))] = TAGConv(3*dim_embed, dim_embed, K = K, normalize = False)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)
            else:
                
                dict_ff[('L'+str(0), '-', 'L'+str(0))] = TAGConv(dim_embed, dim_embed, K = K, normalize = False)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)

            conv_ff = HeteroConv(dict_ff, aggr='sum')

            self.convs.append(conv_ff)
            
            normalizations.append(torch_geometric.nn.norm.InstanceNorm(dim_embed))
            
        self.linear_out  = Linear(dim_embed, dim_embed)
        self.linear_out_coarse  = Linear(dim_embed, dim_embed)
        
        self.normaliz = normalizations
        
        self.edge_model  = EdgeModel(dim_embed*2, [dim_embed, int(dim_embed/2), int(dim_embed/4)], 1)
        self.edge_model_R  = EdgeModel(dim_embed*2, [dim_embed, int(dim_embed/2), int(dim_embed/4)], 1)

        # self.network = torch_geometric.nn.Sequential(self.convs)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr = lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.device = torch.device('cpu')
        self.to(self.device)
        
    def make_graph(self, all_x, all_edge_index, all_edge_attr, grid):
        
        A = grid.A
        graphs = {}

        x = {}
        edge_index = {}
        edge_attr = {}
        idx = {}
        fine2coarse = {}
        edge_attr = {}
        dict_A = {}
        dict_A[0] = A
        x[0], edge_index[0], edge_attr[0] = all_x, all_edge_index, all_edge_attr
        graphs[0] = Data(x[0], edge_index[0], edge_attr[0])

        for i in range(1,self.lvl):
            

            x[i], edge_index[i], idx[i], fine2coarse[i], edge_attr[i], dict_A[i] = K_means_agg_torch(x[i-1], dict_A[i-1], self.ratio, grid)
                        
            graphs[i] = Data(x[i], edge_index[i], edge_attr[i]).to(device)
            
        between_edges = {}

        for i in range(self.lvl-1):
            

            between_edges[str(i)+ ' -> '+str(i+1)] = fine2coarse[i+1]
            between_edges[str(i+1)+ ' -> '+str(i)] = torch.tensor([fine2coarse[i+1][1].tolist(), fine2coarse[i+1][0].tolist()]).to(device)
            

        data = HeteroData()

        for i in range(self.lvl):
            
            data['L'+str(i)].x = x[i].to(device)
            data['L'+str(i),'-','L'+str(i)].edge_index = edge_index[i].to(device)
            data['L'+str(i),'-','L'+str(i)].edge_attr = edge_attr[i].to(device)
            
            if i != self.lvl-1:
                data['L'+str(i),'->','L'+str(i+1)].edge_index = between_edges[str(i)+ ' -> '+str(i+1)]
                data['L'+str(i+1),'->','L'+str(i)].edge_index = between_edges[str(i+1)+ ' -> '+str(i)]
                
                # data['L'+str(i),'->','L'+str(i+1)].edge_attr = self.attr_FC(torch.cat((x[i][between_edges[str(i)+ ' -> '+str(i+1)][0]], 
                #                                                                        x[i+1][between_edges[str(i)+ ' -> '+str(i+1)][1]]), dim =1))
                
                # data['L'+str(i+1),'->','L'+str(i)].edge_attr = self.attr_FC(torch.cat((x[i+1][between_edges[str(i+1)+ ' -> '+str(i)][0]], 
                #                                                                        x[i][between_edges[str(i+1)+ ' -> '+str(i)][1]]), dim =1))

        
        return data.x_dict, data.edge_index_dict, data.edge_attr_dict
        
        
    def forward(self, data, grid):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        edge_attr = self.pre_edge_main(edge_attr)
        
        x_dict, edge_index_dict,  edge_attr_dict = self.make_graph(x, edge_index, edge_attr, grid)

        for key in x_dict.keys():

            x_dict[key] = self.pre_node(x_dict[key])

        for key in edge_attr_dict.keys():
            edge_attr_dict[key] = self.pre_edge(edge_attr_dict[key])

        x_ff = {key: x for key, x in x_dict.items()}
        
        for conv, conv_fc, conv_cf, normalization in zip(self.convs, self.f2c, self.c2f, self.normaliz):
            
            if self.lvl > 1:
                x_fc = conv_fc(x_ff, edge_index_dict)#, edge_attr_dict)
                for key in x_fc.keys():

                    x_fc[key] = x_fc[key].relu()
                

                x_cf = conv_cf(x_ff, edge_index_dict)#, edge_attr_dict)

                for key in x_cf.keys():

                    x_cf[key] = x_cf[key].relu()
                    
                x_ff_new = {}
            
                for i in range(self.lvl):
    
                    if i == 0:

                        x_ff_new['L0'] = torch.cat((x_ff['L0'], x_cf['L0']), dim = 1)
                    
                    elif i == self.lvl-1:
                        
                        x_ff_new['L'+str(i)] = torch.cat((x_ff['L'+str(i)], x_fc['L'+str(i)]), dim = 1)
                        
                    else:
                        
                        x_ff_new['L'+str(i)] = torch.cat((x_fc['L'+str(i)], x_ff['L'+str(i)], x_cf['L'+str(i)]), dim = 1)
    
                x_ff = x_ff_new

            x_ff = conv(x_ff, edge_index_dict, edge_attr_dict)

            for key in x_ff.keys():

                x_ff[key] = normalization(x_ff[key].relu())
            
            
            
        x_coarse = self.linear_out_coarse(x_ff['L1'])
        x = self.linear_out(x_ff['L0'])
        
        row_coarse = edge_index_dict[('L1', '->', 'L0')][0].tolist()
        col_fine = edge_index_dict[('L1', '->', 'L0')][1].tolist()
        
        edge_attr_R = self.edge_model_R(x_coarse[row_coarse], x[col_fine])

        out_R = torch.sparse_coo_tensor([row_coarse, col_fine], edge_attr_R.flatten(),
                                                      (grid.R0.shape[0], grid.R0.shape[1])).double()
        
        # out_R = out_R.to_dense()/out_R.to_dense().sum(0)

        # out_R = out_R.to_sparse()
        
        row = np.array(grid.mask_edges)[:,0].tolist()
        col = np.array(grid.mask_edges)[:,1].tolist()
        
        edge_attr = self.edge_model(x[row], x[col])#, edge_attr.unsqueeze(1)) #+self.edge_model(x[col], x[row], edge_attr_i)

        # out =  edge_attr  #torch.nn.functional.relu(edge_attr) # torch.nn.functional.leaky_relu(edge_attr)
        sz = grid.gdata.x.shape[0]
        out = torch.sparse_coo_tensor([row, col], edge_attr.flatten(),(sz, sz)).to_dense().double()
        # print(out.shape)
        out0 = torch.zeros((out.shape[0], out.shape[0])).double()
        # print(out0.shape)
        return out, out_R# + torch.sparse_coo_tensor([grid.R0.nonzero()[0].tolist(), grid.R0.nonzero()[1].tolist()], 
                                                        #  torch.tensor(grid.R0.data).float(),
                                                        # (grid.R0.shape[0], grid.R0.shape[1])).double()

