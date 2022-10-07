# import matplotlib.pyplot as plt
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
# mpl.rcParams['figure.dpi'] = 300
from ST_CYR import *
import numpy as np
import scipy as sp
from pyamg import amg_core
import numml.sparse as spml

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def match_sparsiy(output, grid):

    sz = grid.gdata.x.shape[0]
    out = torch.sparse_coo_tensor(grid.gdata.edge_index.tolist(), output.flatten(),(sz, sz)).to_dense()
    mask = torch.tensor(grid.gmask.toarray())

    return out * mask

def get_Li (masked, grid):

    L_i = {}
    L = masked

    for i in range(grid.aggop[0].shape[-1]):

        nz = grid.list_cut_aggs[i]

        learnables = grid.learn_nodes[i]


        L_i[i] = torch.zeros(len(nz),len(nz)).double().to(device)

        list_idx = []

        for l in learnables:
            list_idx.append(nz.index(l))

        L_i[i][np.ix_(list_idx, list_idx)] = L[np.ix_(learnables, learnables)]


    return L_i


def coarse_g(grid, ratio):

    mesh = structured(2,2).mesh

    centers = grid.aggop[1].tolist()

    mesh.X = grid.mesh.X[centers]
    mesh.Y = grid.mesh.Y[centers]
    mesh.V = grid.mesh.V[centers]
    mesh.nv = mesh.X.shape[0]

    R0 = grid.aggop[0].transpose()
    A0 = (R0 @ grid.A @ R0.transpose())
    graph = nx.from_scipy_sparse_matrix(A0, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)

    edges = list(graph.edges)
    for i in range(A0.shape[0]):
        edges.remove((i,i))
    E = []
    for i in range(0, len(edges) - 1):
        for j in range(i + 1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            if e1[0] == e2[0]:
                if (e1[1], e2[1]) in edges:
                    E.append([e1[0] , e1[1] , e2[1]])
            else:
                break

    mesh.N = [i for i in range(mesh.X.shape[0])]
    mesh.E = np.array(E)
    mesh.e = edges
    mesh.num_edges = len(mesh.e)
    mesh.ne = len(mesh.E)

    c_g = Grid_PWA(A0, mesh, ratio = ratio, hops = grid.hops, interior = grid.interior,
                       cut=grid.cut, h = grid.h, nu = grid.nu, BC = grid.BC)

    return c_g

softmax = torch.nn.Softmax(dim=0)

def make_sparse_torch(A, sparse = True):
    if sparse:
        idxs = torch.tensor(np.array(A.nonzero()))
        dat = torch.tensor(A.data)
    else:
        idxs = torch.tensor([[i//A.shape[1] for i in range(A.shape[0]*A.shape[1])],
                             [i% A.shape[1] for i in range(A.shape[0]*A.shape[1])]])
        dat = A.flatten()
    s = torch.sparse_coo_tensor(idxs, dat, (A.shape[0], A.shape[1]))
    return s#.to_sparse_csr()


def preconditioner(grid, output, train = False, precond_type = False, u = None):

    if precond_type == 'ML_ORAS':
        t0 = time.time()

    output, out_R0 = output

    M = 0
    tsA = spml.SparseCSRTensor(grid.A).to(device)
    # print (tq5-tq4)
    if precond_type == 'AS':
        for i in range(grid.aggop[0].shape[-1]):

            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()))
            A_inv = torch.linalg.pinv(torch.tensor(grid.R[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R[i].transpose().toarray()))
            M += torch.tensor(grid.R[i].transpose().toarray()) @ A_inv @ torch.tensor(grid.R[i].toarray())
        M = torch.tensor(M)

    elif precond_type == 'RAS':

        for i in range(grid.aggop[0].shape[-1]):

            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            A_inv = torch.linalg.pinv((make_sparse_torch(grid.R_hop[i]) @ make_sparse_torch(grid.A) @ make_sparse_torch(grid.R_hop[i]).t()).to_dense())

            # A_inv = torch.linalg.pinv(torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray()))

            r0 = grid.R[i].nonzero()[-1].tolist()

            rdelta = grid.R_hop[i].nonzero()[-1].tolist()
            list_ixs = []

            for e in r0:
                list_ixs.append(rdelta.index(e))

            modified_R_i = grid.modified_R[i]
            ####

            a1 = scipy.sparse.coo_matrix(modified_R_i).transpose()
            a2 = scipy.sparse.coo_matrix(grid.R_hop[i])

            rows = a1.row.tolist()
            cols = a2.col.tolist()
            row_idx = []
            col_idx = []
            for r in rows:
                for _ in range(len(cols)):
                    row_idx.append(r)

            for _ in range(len(rows)):
                for c in cols:
                    col_idx.append(c)


            list_ixs.sort()

            add_M = torch.sparse_coo_tensor(torch.tensor([row_idx, col_idx]), A_inv[list_ixs, :].flatten(), (grid.A.shape[0], grid.A.shape[1]))
            # add_M = torch.zeros(grid.A.shape).double()
            # add_M[np.ix_(rows, cols)] = A_tilde_inv[list_ixs, :]
            #torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())
            if i == 0:
                M = add_M
            else:
                M += add_M


    elif precond_type == 'ML_ORAS':
        # masked = match_sparsiy(output, grid)
        # t0 = time.time()
        masked = output
        L = get_Li (masked, grid)

        for i in range(grid.aggop[0].shape[-1]):


            r0 = grid.R[i].nonzero()[-1].tolist()
            rdelta = grid.R_hop[i].nonzero()[-1].tolist()
            list_ixs = []
            for e in r0:
                list_ixs.append(rdelta.index(e))

            modified_R_i = grid.modified_R[i]

            modified_L = L[i].to(device)
            grid_Rhop_i = spml.SparseCSRTensor(grid.R_hop[i]).to(device)
            AA =  grid_Rhop_i @ tsA @ grid_Rhop_i.T  ####SPSPMM

            # AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
            A_tilde_inv = torch.linalg.pinv((AA + (1/(grid.h**2))).to_dense()*modified_L)
            # add_M = make_sparse_torch(scipy.sparse.csr_matrix(modified_R_i)).t() @ make_sparse_torch(A_tilde_inv, False) @ make_sparse_torch(grid.R_hop[i])
            # M += add_M.to_dense()

            a1 = scipy.sparse.coo_matrix(modified_R_i).transpose()
            a2 = scipy.sparse.coo_matrix(grid.R_hop[i])

            rows = a1.row.tolist()
            cols = a2.col.tolist()
            row_idx = []
            col_idx = []
            for r in rows:
                for _ in range(len(cols)):
                    row_idx.append(r)

            for _ in range(len(rows)):
                for c in cols:
                    col_idx.append(c)


            list_ixs.sort()

            add_M = torch.sparse_coo_tensor(torch.tensor([row_idx, col_idx]).to(device), A_tilde_inv[list_ixs, :].flatten(), (grid.A.shape[0], grid.A.shape[1])).to(device)
            # add_M = torch.zeros(grid.A.shape).double()
            # add_M[np.ix_(rows, cols)] = A_tilde_inv[list_ixs, :]
            #torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())
            if i == 0:
                M = add_M
            else:
                M += add_M
    else:
        raise RuntimeError('Wrong type for preconditioner: '+str(precond_type))

    if precond_type == 'ML_ORAS':
        t1 = time.time()

    # return M.to_dense(), M.to_dense()
    # t1 = time.time()
    if train:
        M = spml.SparseCSRTensor(M)

        if precond_type == 'ML_ORAS' and out_R0!=None:
            # R0 = out_R0.to_dense().to(device)
            R0 = spml.SparseCSRTensor(out_R0)
        else:
            R0 = torch.tensor(grid.R0.toarray()).to(device)#(grid.aggop[0]*1.0).transpose()
        # R0 = torch.tensor((grid.aggop[0]*1.0).transpose().toarray()).to(device)

        # A0 = R0 @ torch.tensor(grid.A.toarray()).to(device) @ R0.t()
        A0 = R0 @ tsA @ R0.T
        A0_inv = torch.linalg.pinv(A0.to_dense())
        A0_inv_sp = spml.SparseCSRTensor(A0_inv)
        CGC = R0.T @ spml.SparseCSRTensor(A0_inv) @ R0
        # CGC = make_sparse_torch(mul)
        # eye = torch.sparse_coo_tensor(torch.tensor([[jj for jj in range(tsA.shape[0])], [jj for jj in range(tsA.shape[0])]]),
        #                               torch.tensor([1 for jj in range(tsA.shape[0])]), (tsA.shape[0], tsA.shape[1])).double().to_sparse_csr().to(device)
        # eye = torch.eye(tsA.shape[0]).to(device)
        eye = spml.eye(tsA.shape[0]).to(device)
        right_term = eye + (- M @ tsA)
        left_term  = eye + (- CGC @ tsA)
        M_2l = left_term @ right_term
    else:

        if precond_type == 'ML_ORAS' and out_R0!=None:
            R0 = out_R0.to_dense().numpy()
            R0 = sp.sparse.csr_matrix(R0)
        else:
            R0 = grid.R0

        # R0 = grid.R0#(grid.aggop[0]*1.0).transpose()
        # R0 = (grid.aggop[0]*1.0).transpose()
        A0 = R0 @ grid.A @ R0.transpose()
        A0_inv = scipy.sparse.linalg.inv(A0).tocsr()
        CGC = R0.transpose() @ A0_inv @ R0
        eye = scipy.sparse.eye(tsA.shape[0]).tocsr()
        M_2l = scipy.sparse.csr_matrix(M.detach().to_dense().numpy())
        M_2l = (eye - CGC @ grid.A) @ (eye - M_2l @ grid.A)
        M_2l = torch.tensor(M_2l.toarray())
    # t2 = time.time()
    # print(f't10 = {t1-t0}\n')
    # print(f't21 = {t2-t1}\n')
    # sys.exit()
    if precond_type == 'ML_ORAS':
        t2 = time.time()
        print(f't10 = {t1-t0}\n')
        print(f't21 = {t2-t1}\n')
    return M_2l, M


def R0_PoU(grid):

    num_nodes  = grid.aggop[0].shape[0]
    num_aggs = grid.aggop[0].shape[-1]
    R0 = np.zeros((num_aggs, num_nodes))
    for i in range(grid.aggop[0].shape[-1]):
        nzs = grid.R_hop[i].nonzero()[-1].tolist()
        R0[i][nzs] = 1

    return R0/R0.sum(0)


def stationary(grid, out, u = None, K = None, precond_type = 'ML_ORAS'):

    M, _ = preconditioner(grid, out, train = True, precond_type = precond_type, u = u)

    eprop = M

    list_l2 = []
    out_lmax = copy.deepcopy(u)
    for k in range(K):
        out_lmax = eprop @ out_lmax
        l2 = torch.norm(out_lmax, p='fro', dim = 0)
        list_l2.append(l2)

    conv_fact = list_l2[-1]#(list_l2[-1]/list_l2[-3])**0.5
    L_max = torch.dot(softmax(conv_fact), conv_fact)

    return L_max


def stationary_max(grid, out, u = None, K = None, precond_type = 'ML_ORAS'):
    # t0 = time.time()
    eprop, _ = preconditioner(grid, out, train = True, precond_type = precond_type, u = u)
    # t1 = time.time()
    # return torch.norm(eprop)

    list_l2 = []
    out_lmax = copy.deepcopy(u)
    list_max = torch.zeros(K).to(device)
    # list_max_2 = torch.zeros(K)
    for k in range(K):
        out_lmax = eprop @ out_lmax
        l2 = torch.norm(out_lmax, p='fro', dim = 0)

        # list_max_2[k] = max(l2)
        list_max[k] = max(l2) ** (1/(k+1))
        list_l2.append(l2)
    # t2 = time.time()


    conv_fact = list_l2[-1]#(list_l2[-1]/list_l2[-3])**0.5
    # L_max = max(conv_fact)#torch.dot(softmax(conv_fact), conv_fact)
    L_max = (torch.softmax(list_max[2:], dim = 0) * list_max[2:]).sum()#max(list_max)
    # L_max = (list_max_2[-1]/list_max_2[-5]) ** 0.25
    # print(f'The Normalized List = {np.round(np.array(list_max.detach()),2)} \n')
    # print(f'The actual List = {np.round(np.array(list_max_2.detach()),2)} \n')

    # print('****************')
    return L_max


    # out_ras_I, out_ras_R = out
    # out_ras_I = torch.zeros_like(out_ras_I)
    # out_ras_R = None
    # out_RAS = out_ras_I, out_ras_R



    # eprop_RAS, _ = preconditioner(grid, out_RAS, train = True, precond_type = 'ML_ORAS', u = u)

    # # return torch.norm(eprop)

    # list_l2_RAS = []
    # out_lmax_RAS = copy.deepcopy(u)
    # for k in range(K):
    #     out_lmax_RAS = eprop_RAS @ out_lmax_RAS
    #     l2_RAS = torch.norm(out_lmax_RAS, p='fro', dim = 0)
    #     list_l2_RAS.append(l2_RAS)

    # # L_max_list = torch.zeros(K)

    # # for k in range(K):
    # #     L_max_list[k] = (max(list_l2[k])/max(list_l2_RAS[k])) ** 1/(k+1)

    # # sftmx = torch.softmax(L_max_list, dim = 0)
    # # L_max = (L_max_list * sftmx).sum()


    # conv_fact_RAS = list_l2_RAS[-1]#(list_l2[-1]/list_l2[-3])**0.5
    # L_max_RAS = max(conv_fact_RAS)#torch.dot(softmax(conv_fact), conv_fact)

    # L_max = L_max/L_max_RAS



    # return L_max



def test_stationary(grid, out, precond_type, u, K, M=None):

    if M is None:
        M, _ = preconditioner(grid, out, train = False, precond_type = precond_type, u = u)

    eprop_a = M

    out = copy.deepcopy(u)
    l2_list = []
    l2 = torch.norm(out, p='fro', dim = 0)
    l2_list.append(torch.dot(softmax(l2), l2))
    for k in range(K):
        out = eprop_a @ out
        l2 = torch.norm(out, p='fro', dim = 0)
        l2_list.append(torch.dot(softmax(l2), l2))

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
