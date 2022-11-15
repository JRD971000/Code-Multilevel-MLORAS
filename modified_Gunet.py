#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:10:33 2022

@author: alitaghibakhshi
"""

import torch
import torch.nn.functional as F
from torch_sparse import spspmm
from torch.nn import Linear, ReLU
from torch.nn.functional import relu, sigmoid
from torch_geometric.nn import GCNConv, TopKPooling, TAGConv
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    sort_edge_index,
)
from torch_geometric.utils.repeat import repeat
import torch.optim as optim
import torch_geometric
import sys

class modified_GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, lvl, K=2, 
                 pool_ratios=0.5, sum_res=True, act=F.relu, lr = 1e-3):
        super().__init__()
        depth = lvl-1
        assert depth >= 1
        self.name = "GUnet_depth"+str(depth)+"_num_level"+str(num_layers)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res
        self.num_layers = num_layers

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.down_features = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        self.up_features = torch.nn.ModuleList()
        self.normalize_down = torch.nn.ModuleList()
        self.normalize_up = torch.nn.ModuleList()

        for layer in range(num_layers):
            in_channels = self.in_channels
            # self.down_convs.append(GCNConv(in_channels, channels, improved=True, normalize=False))
            
            self.down_convs.append(TAGConv(in_channels, channels, K = K, normalize = False))
            self.down_features.append(torch.nn.Sequential(Linear(channels, channels), 
                                                          ReLU(), Linear(channels,channels)))
            for i in range(depth):
                self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
                # self.down_convs.append(GCNConv(channels, channels, improved=True, normalize=False))
                self.down_convs.append(TAGConv(channels, channels, K = K, normalize = False))
                self.down_features.append(torch.nn.Sequential(Linear(channels, channels), 
                                                              ReLU(), Linear(channels,channels)))
                
                self.normalize_down.append(torch_geometric.nn.norm.InstanceNorm(channels))
                self.normalize_up.append(torch_geometric.nn.norm.InstanceNorm(channels))

            in_channels = channels if sum_res else 2 * channels
    
            
            for i in range(depth - 1):
                # self.up_convs.append(GCNConv(in_channels, channels, improved=True, normalize=False))
                self.up_convs.append(TAGConv(in_channels, channels, K = K, normalize = False))
                self.up_features.append(torch.nn.Sequential(Linear(channels, channels), 
                                                              ReLU(), Linear(channels,channels)))
            # self.up_convs.append(GCNConv(in_channels, out_channels, improved=True, normalize=False))
            
            self.up_convs.append(TAGConv(in_channels, channels, K = K, normalize = False))
            self.up_features.append(torch.nn.Sequential(Linear(channels, channels), 
                                                          ReLU(), Linear(channels,1)))
        
        # self.out_fc = torch.nn.Sequential(Linear(1, channels), ReLU(), 
        #                             Linear(channels, channels), ReLU(),
        #                             Linear(channels, 1))
        # self.reset_parameters()
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr = lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.device = torch.device('cpu')
        self.to(self.device)

    # def reset_parameters(self):
    #     for conv in self.down_convs:
    #         conv.reset_parameters()
    #     for pool in self.pools:
    #         pool.reset_parameters()
    #     for conv in self.up_convs:
    #         conv.reset_parameters()


    def forward(self, data, A, batch=None):
        """"""
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.flatten()
        
        # edge_weight = x.new_ones(edge_index.size(1))

        for layer in range(self.num_layers):
            
            batch = edge_index.new_zeros(x.size(0))

            x = self.down_convs[layer * (self.depth+1)](x, edge_index, edge_weight)
            # print("****", layer, x.shape, edge_index.shape, edge_weight.shape)
            x = self.act(x)
    
            xs = [x]
            edge_indices = [edge_index]
            edge_weights = [edge_weight]
            perms = []
            
            for i in range(1, self.depth + 1):
                edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                           x.size(0))


                x, edge_index, edge_weight, batch, perm, _ = self.pools[layer * (self.depth) + i - 1]( 
                    x, edge_index, edge_weight, batch)
                
                x = self.down_convs[layer * (self.depth+1) + i](x, edge_index, edge_weight)
                x = self.down_features[layer * (self.depth+1) + i](x)
                # x = self.normalize_down[layer * self.depth + i-1](x) 
                x = self.act(x)
    
                if i < self.depth:
                    xs += [x]
                    edge_indices += [edge_index]
                    edge_weights += [edge_weight]
                perms += [perm]
    
            
            
            for i in range(self.depth):
                j = self.depth - 1 - i
    
                res = xs[j]
                edge_index = edge_indices[j]
                edge_weight = edge_weights[j]
                perm = perms[j]
    
                up = torch.zeros_like(res)
                up[perm] = x
                x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
    
                x = self.up_convs[layer * self.depth + i](x, edge_index, edge_weight)
                x = self.up_features[layer * self.depth + i](x)
                x = self.normalize_up[layer * self.depth + i](x) 
                x = self.act(x) if i < self.depth - 1 else x
            
        
        # x = self.out_fc(x)
            
        self.perm = perm
        return x


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
