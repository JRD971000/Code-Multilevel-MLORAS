import torch as T
import torch
import torch_geometric
import torch.optim as optim
from torch.nn import ReLU, GRU, Sequential, Linear
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn as nn
from torch_geometric.nn import (NNConv, GATConv, graclus, max_pool, max_pool_x, 
                                global_mean_pool, BatchNorm, InstanceNorm, GraphConv,
                                GCNConv, TAGConv, SGConv, LEConv, TransformerConv, SplineConv,
                                GMMConv, GatedGraphConv, ARMAConv, GENConv, DeepGCNLayer,
                                LayerNorm, GraphUNet, ChebConv)
from torch.nn.functional import relu, sigmoid
import numpy as np
import sys

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
        
class Brute_Force(T.nn.Module):
    def __init__(self, out, lr):
        super(Brute_Force, self).__init__()
        self.FC = Linear(1, out)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
         
        self.device = T.device('cpu')
        self.to(self.device)
        
    def forward(self):
        
        x = self.FC(torch.tensor([1]).float())
        
        return x
    
        
class FC_test(T.nn.Module):
    def __init__(self, inp, out, dim, lr):
        super(FC_test, self).__init__()
        
        self.FC1 = Linear(inp, dim)
        self.FC2 = Linear(dim, dim)
        self.FC3 = Linear(dim, dim)
        self.FC4 = Linear(dim, out)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = T.device('cpu')
        self.to(self.device)
        
    def forward(self, D, grid):
        
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        x = x.flatten()
        x = relu(self.FC1(x))
        x = relu(self.FC2(x))
        x = relu(self.FC3(x))
        x = self.FC4(x)
        
        row = np.array(grid.mask_edges)[:,0].tolist()
        col = np.array(grid.mask_edges)[:,1].tolist()
        
        sz = grid.gdata.x.shape[0]
        out = torch.sparse_coo_tensor([row, col], x.flatten(), (sz, sz)).to_dense().double()
        
        return out
class mloras_net(T.nn.Module):
    
    def __init__(self, dim = 128, K = 2, num_res = 8, num_convs = 4, lr = 0.0001, res = True, tf = None):
        
        super(mloras_net, self).__init__()
        
        self.dim = dim
        self.K = K
        self.num_res = num_res
        self.num_convs = num_convs
        self.res = res
        self.tf = tf
        
        conv_blocks = []
        feature_blocks = []
        param_block = []
        normalizations = []
        
        
        self.FC1 = Linear(2, int(dim/4))
        self.FC2 = Linear(int(dim/4), int(dim/2))
        self.FC3 = Linear(int(dim/2), int(dim/4))
        self.FC4_TAG  = Linear(int(dim/4), 1)
        self.normalize_attr = torch_geometric.nn.norm.InstanceNorm(int(dim/4))
        
        
        
        if res:
            for i in range(num_convs):
                if i == 0:
                    conv_block = [(TAGConv(1, dim, K=K, normalize = False), 'x, edge_index, edge_attr -> x')]
                else:
                    conv_block = [(TAGConv(dim, dim, K=K, normalize = False), 'x, edge_index, edge_attr -> x')]

                param_block.append(*conv_block)
                conv_blocks.append(torch_geometric.nn.Sequential('x, edge_index, edge_attr', conv_block))
                
                feature_block = FeatureResNet(dim, [dim for k in range(num_res+1)], dim)
                feature_blocks.append(feature_block)
                param_block.append(feature_block.network)
                
                normalizations.append(torch_geometric.nn.norm.InstanceNorm(dim))
            
        else:
            for i in range(num_convs):

                if i == 0:
                    conv_block = [(TAGConv(1, dim, K=K, normalize = False), 'x, edge_index, edge_attr -> x')]
                else:
                    conv_block = [(TAGConv(dim, dim, K=K, normalize = False), 'x, edge_index, edge_attr -> x')]
                param_block.append(*conv_block)
                conv_blocks.append(torch_geometric.nn.Sequential('x, edge_index, edge_attr', conv_block))
                
                
                normalizations.append(torch_geometric.nn.norm.InstanceNorm(dim))
            
        if tf:
            
            self.transformer = torch.nn.Transformer(nhead=8, batch_first=True, d_model=dim)
            
        self.edge_model  = EdgeModel(dim*2, [dim, int(dim/2), int(dim/4)], 1)


        self.feature_blocks = feature_blocks
        self.conv_blocks = conv_blocks
        self.normaliz = normalizations
        self.network = torch_geometric.nn.Sequential('x, edge_index, edge_attr', param_block)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr = lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.device = T.device('cpu')
        self.to(self.device)
        
        
    def forward(self, D, grid):
    
        x, edge_index, edge_attr_i = D.x, D.edge_index, D.edge_attr
        
        edge_attr = relu(self.FC1(edge_attr_i))
        edge_attr = relu(self.FC2(edge_attr))
        edge_attr = self.normalize_attr(relu(self.FC3(edge_attr)))
        edge_attr = self.FC4_TAG(edge_attr).flatten()
                
        # row = edge_index[0]
        # col = edge_index[1]
        
        row = np.array(grid.mask_edges)[:,0].tolist()
        col = np.array(grid.mask_edges)[:,1].tolist()
        
        edge_attr = edge_attr.flatten()                            #uncomment for TAG conv
        
        # identity = x
        if self.res:
            
            for conv_block, feature_block, normalization in zip(self.conv_blocks, self.feature_blocks, self.normaliz):

                x = relu(conv_block(x, edge_index, edge_attr))
                x = normalization(x)
                x = relu(feature_block(x))
            
        else:

            for conv_block, normalization in zip(self.conv_blocks, self.normaliz):
                
                x = relu(conv_block(x, edge_index, edge_attr))
                x = normalization(x)

        
        if self.tf:
            '''
            x = self.transformer(x, x)
            '''
            max_len = int(max(grid.aggop[0].sum(0).tolist()[0]))
            num_aggs = grid.aggop[0].shape[-1]

            tf_input = torch.zeros((num_aggs, max_len, self.dim))
            padding_mask  = torch.ones((num_aggs, max_len))

            for i in range(num_aggs):
                
                subd_feature = x[grid.list_aggs[i]]
                tf_input[i, 0:subd_feature.shape[0], :] = subd_feature
                padding_mask[i, 0:subd_feature.shape[0]] = 0
                
            padding_mask = padding_mask.bool()

            out = self.transformer(tf_input, tf_input, src_key_padding_mask = padding_mask,
                                    tgt_key_padding_mask = padding_mask, memory_key_padding_mask=padding_mask)

            x_ = torch.zeros_like(x)
            for i in range(num_aggs):
                
                x_[grid.list_aggs[i], :] = out[i, :len(grid.list_aggs[i]),:]
                
            x = x_
            
        edge_attr = self.edge_model(x[row], x[col])#, edge_attr.unsqueeze(1)) #+self.edge_model(x[col], x[row], edge_attr_i)

        # out =  edge_attr  #torch.nn.functional.relu(edge_attr) # torch.nn.functional.leaky_relu(edge_attr)
        sz = grid.gdata.x.shape[0]
        out = torch.sparse_coo_tensor([row, col], edge_attr.flatten(),(sz, sz)).to_dense().double()
        
        return out#, minv
  
    
########################Implementing Resnet###############################

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class FeatureResNet(nn.Module):
    def __init__(self, in_dim, dims, out_dim):
        super(FeatureResNet, self).__init__()

        blocks = nn.ModuleList([])
        activations = nn.ModuleList([])
        normalizations = nn.ModuleList([])
        param_block = nn.ModuleList([])
        
        # this_block = nn.ModuleList([Lin(in_dim, dims[0])])
        # param_block.extend(this_block)
        # blocks.append(Seq(*this_block))
        # activations.append(activation_func('relu'))
        # normalizations.append(nn.LayerNorm(dims[0]))
        # normalizations.append(activation_func('none'))
        
        
        for i in range(len(dims)-1):
            
            this_block = nn.ModuleList([Lin(dims[i], dims[i+1]), ReLU(), Lin(dims[i+1], dims[i+1])])

            param_block.extend(this_block)
            blocks.append(Seq(*this_block))
            activations.append(activation_func('relu'))
            normalizations.append(nn.LayerNorm(dims[i+1]))
            # normalizations.append(activation_func('none'))
        
        # this_block = nn.ModuleList([Lin(dims[-1], out_dim)])
        # param_block.extend(this_block)
        # blocks.append(Seq(*this_block))

        # activations.append(activation_func('none'))
        # normalizations.append(activation_func('none'))
        
        self.blocks = blocks
        self.activate = activations
        self.normaliz = normalizations
        self.network = Seq(*param_block)
        
        
    def forward(self, x):

        for block, activate, normalization in zip(self.blocks, self.activate, self.normaliz):
            residual = x
            if normalization is not None:
                
                x = normalization(block(x))
                
            else:
                
                x = block(x)

            if x.shape == residual.shape:

                x += residual

            x = activate(x)
            
        return x
    
    

    
class EdgeModel(nn.Module):
    
    def __init__(self, in_dim, dims, out_dim):
        super(EdgeModel, self).__init__()

        blocks = nn.ModuleList([])
        activations = nn.ModuleList([])
        normalizations = nn.ModuleList([])
        param_block = nn.ModuleList([])
        
        
        this_block = nn.ModuleList([Lin(in_dim, dims[0])])
        
        param_block.extend(this_block)
        blocks.append(Seq(*this_block))
        activations.append(activation_func('relu'))
        normalizations.append(activation_func('none'))
        
        
        for i in range(len(dims)-1):
            
            this_block = nn.ModuleList([Lin(dims[i], dims[i+1]), ReLU(), Lin(dims[i+1], dims[i+1])])

            param_block.extend(this_block)
            blocks.append(Seq(*this_block))
            activations.append(activation_func('relu'))
            # normalizations.append(activation_func('none'))
            normalizations.append(nn.LayerNorm(dims[i+1]))

        this_block = nn.ModuleList([Lin(dims[-1], out_dim)])

        param_block.extend(this_block)
        blocks.append(Seq(*this_block))

        activations.append(activation_func('none'))
        normalizations.append(activation_func('none'))
        # normalizations.append(nn.LayerNorm(out_dim))
        
        self.blocks = blocks
        self.activate = activations
        self.normaliz = normalizations
        self.network = Seq(*param_block)
        
        
    def forward(self, src, dest):#, edge_attr):#, u, batch):
        
        x = torch.cat([src, dest], 1)#, edge_attr],1)#, u[batch]], 1)
        
        for block, activate, normalization in zip(self.blocks, self.activate, self.normaliz):
            residual = x
            if normalization is not None:

                x = normalization(block(x))
                
            else:
                
                x = block(x)

            if x.shape == residual.shape:

                x += residual

            x = activate(x)
            
        return x
    
 