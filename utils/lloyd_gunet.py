#!/usr/bin/env python3


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
from NNs import EdgeModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
class PlainMP(MessagePassing):
    def __init__(self, dim_embed, aggr = 'add'):
        super().__init__(aggr=aggr) #  "Max" aggregation.
        self.net = torch.nn.Sequential(Linear(2*dim_embed, dim_embed), 
                                # torch.nn.ReLU(), Linear(dim_embed, dim_embed),
                                torch.nn.ReLU(), Linear(dim_embed, dim_embed))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        
        x = torch.cat((x_i, x_j), dim = 1)
        x = self.net(x)
        return x#x_i

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

def K_means_agg(X, A, ratio):
    
    R, idx = lloyd_aggregation(A, ratio)
    sum_R = scipy.sparse.diags(1/np.array(R.sum(0))[0])
    R = R @ sum_R
    idx = torch.tensor(idx)
    A_coarse = R.transpose() @ A @ R
    coarse_index = torch.tensor(np.int64(A_coarse.nonzero()))
    
    X_coarse = (X.t() @ torch.tensor(R.toarray()).float()).t()
    
    
    fine2coarse = torch.tensor(np.int64(R.nonzero()))
    attr_coarse = torch.tensor(A_coarse.toarray().flatten()[A_coarse.toarray().flatten().nonzero()]).unsqueeze(1).float()
    
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

    
    # fine2coarse = torch.tensor(np.int64(R.nonzero())).to(device)
    
    neigh_R0 = grid.neigh_R0.transpose()
    
    fine2coarse = torch.tensor(np.int64(neigh_R0.nonzero())).to(device)
    

    # fine2coarse = tsR.to_dense().nonzero().to(device)
    # fine2coarse = fine2coarse.reshape(fine2coarse.shape[1], fine2coarse.shape[0])
    
    attr_coarse = torch.tensor(A_coarse.toarray().flatten()[A_coarse.toarray().flatten().nonzero()]).unsqueeze(1).float().to(device)
    # attr_coarse = A_coarse.to_dense().flatten()[A_coarse.to_dense().flatten().nonzero()].float().to(device)

    return X_coarse, coarse_index, idx, fine2coarse, attr_coarse, A_coarse


class lloyd_gunet(torch.nn.Module):
    def __init__(self, lvl, num_layers, dim_embed, K = 2, ratio = 0.5, lr =1e-3):
        super().__init__()
        
        
        self.lvl = lvl
        self.num_layers = num_layers
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
        
        self.convs_down = torch.nn.ModuleList()
        self.down_conv_features = torch.nn.ModuleList()
        self.convs_up = torch.nn.ModuleList()
        self.up_conv_features = torch.nn.ModuleList()

        self.f2c   = torch.nn.ModuleList()
        self.c2f   = torch.nn.ModuleList()
        self.coarsest_conv = torch.nn.ModuleList()

        self.name  = 'lloyd_gunet_lvl'+str(lvl)
        normalizations = torch.nn.ModuleList()

        for layer in range(num_layers):
            for i in range(lvl):
                
     
                dict_ff_down = {}
                dict_ff_up = {}
                dict_coarsest = {}
                dict_fc = {}
                dict_cf = {}
                
                if i != lvl-1:
                    dict_fc[('L'+str(i), '->', 'L'+str(i+1))] = PlainMP(dim_embed)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)#TAGConv(dim_embed, dim_embed, K = 2, normalize = False)#
                    conv_fc = HeteroConv(dict_fc, aggr='add')
                    self.f2c.append(conv_fc)            
    
                if i != 0:
                    dict_cf[('L'+str(self.lvl-i), '->', 'L'+str(self.lvl-1-i))] = PlainMP(dim_embed)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)#TAGConv(dim_embed, dim_embed, K = 2, normalize = False)#
                    conv_cf = HeteroConv(dict_cf, aggr='add')
                    self.c2f.append(conv_cf)
    
                if i!=lvl-1:
                    dict_ff_down[('L'+str(i), '-', 'L'+str(i))] = TAGConv(dim_embed, dim_embed, K = K, normalize = False)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)
                    conv_ff_down = HeteroConv(dict_ff_down, aggr='add')
                    self.convs_down.append(conv_ff_down)
                    # self.down_conv_features.append(Linear(dim_embed, dim_embed))
                if i != 0:
                    dict_ff_up[('L'+str(self.lvl-1-i), '-', 'L'+str(self.lvl-1-i))] = TAGConv(dim_embed, dim_embed, K = K, normalize = False)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)
                    conv_ff_up = HeteroConv(dict_ff_up, aggr='add')
                    self.convs_up.append(conv_ff_up)
                    # self.up_conv_features.append(Linear(dim_embed, dim_embed))
    
            normalizations.append(torch_geometric.nn.norm.InstanceNorm(dim_embed))
            dict_coarsest[('L'+str(self.lvl-1), '-', 'L'+str(self.lvl-1))] = TAGConv(dim_embed, dim_embed, K = K, normalize = False)
            conv_coarsest = HeteroConv(dict_coarsest, aggr='add')
            self.coarsest_conv.append(conv_coarsest)
                
        
        self.normaliz = normalizations
        
        self.linear_out  = Linear(dim_embed, dim_embed)
        self.linear_out_coarse  = Linear(dim_embed, dim_embed)
        
        self.normaliz = normalizations
        
        self.edge_model  = EdgeModel(dim_embed*2, [dim_embed, int(dim_embed/2), int(dim_embed/4)], 1)
        self.edge_model_R  = EdgeModel(dim_embed*2, [dim_embed, int(dim_embed/2), int(dim_embed/4)], 1)
        
        # self.edge_model_R_dense = torch.nn.Sequential(Linear(dim_embed, dim_embed), 
        #                         torch.nn.ReLU(), Linear(dim_embed, dim_embed),
        #                         torch.nn.ReLU(), Linear(dim_embed, 29))

        # self.network = torch_geometric.nn.Sequential(self.convs)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-5)
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
            if i == 1:
                x[i], edge_index[i], idx[i], fine2coarse[i], edge_attr[i], dict_A[i] = K_means_agg_torch(x[i-1], dict_A[i-1], self.ratio, grid)
            else:
                x[i], edge_index[i], idx[i], fine2coarse[i], edge_attr[i], dict_A[i] = K_means_agg(x[i-1], dict_A[i-1], self.ratio)

            graphs[i] = Data(x[i], edge_index[i], edge_attr[i])
            
        between_edges = {}

        for i in range(self.lvl-1):
            

            between_edges[str(i)+ ' -> '+str(i+1)] = fine2coarse[i+1]
            between_edges[str(i+1)+ ' -> '+str(i)] = torch.tensor([fine2coarse[i+1][1].tolist(), fine2coarse[i+1][0].tolist()])
            

        data = HeteroData()

        for i in range(self.lvl):
            
            data['L'+str(i)].x = x[i]
            data['L'+str(i),'-','L'+str(i)].edge_index = edge_index[i]
            data['L'+str(i),'-','L'+str(i)].edge_attr = edge_attr[i]
            
            if i != self.lvl-1:
                data['L'+str(i),'->','L'+str(i+1)].edge_index = between_edges[str(i)+ ' -> '+str(i+1)]
                data['L'+str(i+1),'->','L'+str(i)].edge_index = between_edges[str(i+1)+ ' -> '+str(i)]
                
                # data['L'+str(i),'->','L'+str(i+1)].edge_attr = self.attr_FC(torch.cat((x[i][between_edges[str(i)+ ' -> '+str(i+1)][0]], 
                #                                                                        x[i+1][between_edges[str(i)+ ' -> '+str(i+1)][1]]), dim =1))
                
                # data['L'+str(i+1),'->','L'+str(i)].edge_attr = self.attr_FC(torch.cat((x[i+1][between_edges[str(i+1)+ ' -> '+str(i)][0]], 
                #                                                                        x[i][between_edges[str(i+1)+ ' -> '+str(i)][1]]), dim =1))

        self.perm = idx[1]
        return data.x_dict, data.edge_index_dict, data.edge_attr_dict#, perm
        
        
    def forward(self, data, grid, train):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_attr = self.pre_edge_main(edge_attr)

        x_dict, edge_index_dict,  edge_attr_dict = self.make_graph(x, edge_index, edge_attr, grid)

        for key in x_dict.keys():

            x_dict[key] = self.pre_node(x_dict[key])

        for key in edge_attr_dict.keys():
            edge_attr_dict[key] = self.pre_edge(edge_attr_dict[key])

        x_ff = {key: x for key, x in x_dict.items()}
        
        for layer in range(self.num_layers):
            
            list_level = [i for i in range(self.lvl-1)]
            for i in list_level:
                
                out = self.convs_down[layer*(self.lvl-1) + i](x_ff, edge_index_dict, edge_attr_dict)
                # out_ = self.down_conv_features[layer*(self.lvl-1) + i](out['L'+str(i)])

                x_ff['L'+str(i)] = out['L'+str(i)]
 
                out = self.f2c[layer*(self.lvl-1) + i](x_ff, edge_index_dict)#, edge_attr_dict)
                x_ff['L'+str(i+1)] = out['L'+str(i+1)]
    
            out = self.coarsest_conv[layer](x_ff, edge_index_dict, edge_attr_dict)
            x_ff['L'+str(self.lvl-1)] = out['L'+str(self.lvl-1)]
            
            x_ff = {key: self.normaliz[layer](x) for key, x in x_ff.items()}
            
            list_level = [self.lvl - 1 - i for i in range(self.lvl-1)]
            for j, i in enumerate(list_level):
                
                out = self.c2f[layer*(self.lvl-1) + j](x_ff, edge_index_dict)#, edge_attr_dict)
                x_ff['L'+str(i-1)] = out['L'+str(i-1)]
                out = self.convs_up[layer*(self.lvl-1) + j](x_ff, edge_index_dict, edge_attr_dict)
                # out_ = self.up_conv_features[layer*(self.lvl-1) + j](out['L'+str(i-1)])
                x_ff['L'+str(i-1)] = out['L'+str(i-1)]
   
                
        x_coarse = self.linear_out_coarse(x_ff['L1'])
        x = self.linear_out(x_ff['L0'])
        
        row_coarse = edge_index_dict[('L1', '->', 'L0')][0].tolist()
        col_fine = edge_index_dict[('L1', '->', 'L0')][1].tolist()
        
        
        edge_attr_R = self.edge_model_R(x_coarse[row_coarse], x[col_fine])
        
        out_R = torch.sparse_coo_tensor([row_coarse, col_fine], edge_attr_R.flatten(),
                                                      (grid.R0.shape[0], grid.R0.shape[1])).double()
        

        if train:
            out_R = out_R.to_dense()/out_R.to_dense().sum(0)

            out_R = out_R.to_sparse()
            
        else:
            
            sum_row_mat = 1/torch.sparse.sum(out_R, dim = 0).coalesce().values()
            sum_row_mat = torch.sparse_coo_tensor([np.arange(sum_row_mat.shape[0]).tolist(),
                                                      np.arange(sum_row_mat.shape[0]).tolist()], sum_row_mat, 
                                                      (sum_row_mat.shape[0], sum_row_mat.shape[0])).double().to_sparse_csr()
              
            out_R = sum_row_mat @ out_R.to_sparse_csr().t()
    
            out_R = out_R.t().to_sparse_coo()

        
        row = np.array(grid.mask_edges)[:,0].tolist()
        col = np.array(grid.mask_edges)[:,1].tolist()
        
        edge_attr = self.edge_model(x[row], x[col])#, edge_attr.unsqueeze(1)) #+self.edge_model(x[col], x[row], edge_attr_i)

        # out =  edge_attr  #torch.nn.functional.relu(edge_attr) # torch.nn.functional.leaky_relu(edge_attr)
        sz = grid.gdata.x.shape[0]
        out = torch.sparse_coo_tensor([row, col], edge_attr.flatten(),(sz, sz)).double()#.to_dense()
        # print(out.shape)
        # out0 = torch.zeros((out.shape[0], out.shape[0])).double()
        # print(out0.shape)
        
        if train:
            
            out = out.to_dense()
            
        else:
            
            out = out.to_sparse_csr()
            
        return out, out_R# + torch.sparse_coo_tensor([grid.R0.nonzero()[0].tolist(), grid.R0.nonzero()[1].tolist()], 
                                                        #  torch.tensor(grid.R0.data).float(),
                                                        # (grid.R0.shape[0], grid.R0.shape[1])).double()







# x = {'L0' : torch.ones((6,4)), 'L1' : torch.ones((2,4))}
# edge_index_dict = {('L0','-', 'L0'): torch.tensor([[  0,   0,  0, 2, 1, 4, 5, 3],
#                                                     [  0,   1,  1, 1, 0, 1, 0, 1]]),
                   
#                     ('L0','->', 'L1'): torch.tensor([[  0, 2, 1, 4, 5, 3],
#                                                     [  0, 0, 1, 0, 0, 1]]),
                   
#                     ('L1','->', 'L0'): torch.tensor([[ 0, 0, 1, 0, 1, 1],
#                                                     [  0, 2, 1, 4, 5, 3]]),
                   
#                     ('L1','-', 'L1'): torch.tensor([[0, 1, 1, 1, 0],
#                                                     [1, 1, 0, 1, 0]])}
# edge_attr_dict = {
#                         ('L0','-', 'L0'): torch.rand((8,1)),
                       
#                         # ('L0','->', 'L1'): torch.rand((6,1)),
                       
#                         # ('L1','->', 'L0'): torch.rand((6,1)),
                       
#                         ('L1','-', 'L1'): torch.rand((5,1))
#     }

# hetero_conv = HeteroConv({
#     ('L0', '->', 'L1'): PlainMP('sum')
# }, aggr='sum')

# out = hetero_conv(x, edge_index_dict, edge_attr_dict)

# sys.exit()
 