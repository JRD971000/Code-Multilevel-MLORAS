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


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Brute_Force(T.nn.Module):
    '''
    A brute force optimization over the boundary values directly.
    There is no graph network model here, but rather we are optimizing the problem.
    '''

    def __init__(self, out, lr):
        super(Brute_Force, self).__init__()
        self.FC = Linear(1, out)

        self.optimizer = optim.Adam(self.parameters(), lr = lr)

        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self):
        return self.FC(torch.tensor([1]).float())


class FC_test(T.nn.Module):
    def __init__(self, inp, out, dim, lr):
        super(FC_test, self).__init__()

        self.FC1 = Linear(inp, dim)
        self.FC2 = Linear(dim, dim)
        self.FC3 = Linear(dim, dim)
        self.FC4 = Linear(dim, out)

        self.optimizer = optim.RMSprop(self.parameters(), lr = lr, alpha=0.99,
                            eps=1e-08, weight_decay=0, momentum=0, centered=False)

        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, D):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        x = x.flatten()
        x = relu(self.FC1(x))
        x = relu(self.FC2(x))
        x = relu(self.FC3(x))
        x = self.FC4(x)

        return x


class mloras_net(T.nn.Module):
    '''
    The graph neural network model used to learn the interface conditions
    '''

    def __init__(self, dim = 128, K = 2, num_res = 8, num_convs = 4, lr = 0.0001):
        '''
        Parameters
        ----------
        dim : integer
          Dimensionality of hidden layers
        K : integer
          Number of "hops" for each node convolution
        num_res : integer
          Number of residual layers after each nodal convolution.  Set this to
          zero to disable any sort of residual propagation.
        lr : float
          Learning rate for the Adam optimizer
        '''

        super(mloras_net, self).__init__()

        self.dim = dim
        self.K = K
        self.num_res = num_res
        self.num_convs = num_convs

        conv_blocks = []
        feature_blocks = []
        param_block = []
        normalizations = []

        self.FC1 = Linear(2, int(dim/4))
        self.FC2 = Linear(int(dim/4), int(dim/2))
        self.FC3 = Linear(int(dim/2), int(dim/4))
        self.FC4_TAG  = Linear(int(dim/4), 1)
        self.normalize_attr = torch_geometric.nn.norm.InstanceNorm(int(dim/4))

        # Enable the residual layers if num_res is a positive value
        self.res = (num_res > 0)
        if self.res:
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

        self.edge_model = EdgeModel(dim*2+1, [dim, int(dim/2), int(dim/4)], 1)

        self.feature_blocks = feature_blocks
        self.conv_blocks = conv_blocks
        self.normaliz = normalizations
        self.network = torch_geometric.nn.Sequential('x, edge_index, edge_attr', param_block)

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = T.device('cpu')
        self.to(self.device)


    def forward(self, D, grid):
        x, edge_index, edge_attr_i = D.x, D.edge_index, D.edge_attr

        edge_attr = relu(self.FC1(edge_attr_i))
        edge_attr = relu(self.FC2(edge_attr))
        edge_attr = self.normalize_attr(relu(self.FC3(edge_attr)))
        edge_attr = self.FC4_TAG(edge_attr).flatten()

        row = edge_index[0]
        col = edge_index[1]

        edge_attr = edge_attr.flatten()                            #uncomment for TAG conv

        if self.res:
            for conv_block, feature_block, normalization in zip(self.conv_blocks, self.feature_blocks, self.normaliz):
                x = relu(conv_block(x, edge_index, edge_attr))
                x = normalization(x)
                x = relu(feature_block(x))
        else:
            for conv_block, normalization in zip(self.conv_blocks, self.normaliz):
                x = relu(conv_block(x, edge_index, edge_attr))
                x = normalization(x)

        edge_attr = self.edge_model(x[row], x[col], edge_attr.unsqueeze(1)) #+self.edge_model(x[col], x[row], edge_attr_i)

        sz = grid.gdata.x.shape[0]
        out = torch.sparse_coo_tensor([row.tolist(), col.tolist()], edge_attr.flatten(),(sz, sz)).to_dense().double()
        return out


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

        for i in range(len(dims)-1):
            this_block = nn.ModuleList([Lin(dims[i], dims[i+1]), ReLU(), Lin(dims[i+1], dims[i+1])])

            param_block.extend(this_block)
            blocks.append(Seq(*this_block))
            activations.append(activation_func('relu'))
            normalizations.append(nn.LayerNorm(dims[i+1]))

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
            normalizations.append(nn.LayerNorm(dims[i+1]))

        this_block = nn.ModuleList([Lin(dims[-1], out_dim)])

        param_block.extend(this_block)
        blocks.append(Seq(*this_block))

        activations.append(activation_func('none'))
        normalizations.append(activation_func('none'))

        self.blocks = blocks
        self.activate = activations
        self.normaliz = normalizations
        self.network = Seq(*param_block)


    def forward(self, src, dest, edge_attr):
        x = torch.cat([src, dest, edge_attr],1)

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
