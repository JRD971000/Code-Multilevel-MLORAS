
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
import numml
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

def build_eprop_op(grid, R0, M, model = 'ML_ORAS'):
    
    M = M.to_dense()
    if model == 'RAS':
        
        R0 = make_sparse_torch(R0).to_sparse_csr().to(device).to_dense()
    else:
        R0 = R0.to_sparse_csr().to_dense()
        
    tsA = make_sparse_torch(grid.A).to_sparse_csr().to(device).to_dense()

    A0 = R0 @ tsA @ R0.t()
    A0_inv = torch.linalg.pinv(A0).to(device)

    CGC = R0.t() @ A0_inv @ R0
    I = torch.eye(grid.A.shape[0]).to(device)
    M_out = (I - CGC @ tsA) @ (I- M @ tsA)
    
    return M_out




def precond_cg(grid, R0, M, model = 'ML_ORAS'):
    
    M = M.to_dense()
    if model == 'RAS':
        
        R0 = make_sparse_torch(R0).to_sparse_csr().to(device).to_dense()
    else:
        R0 = R0.to_sparse_csr().to_dense()
        
    tsA = make_sparse_torch(grid.A).to_sparse_csr().to(device).to_dense()

    A0 = R0 @ tsA @ R0.t()
    A0_inv = torch.linalg.pinv(A0).to(device)
    I = torch.eye(grid.A.shape[0]).to(device)

    K = M + R0.t() @ A0_inv @ R0 @ (I-tsA@M)
    prec = K + M.t()@(I-tsA@K)
    
    return prec


# def get_Li (masked, grid):
    
#     L_i = {}
#     L = masked

#     for i in range(grid.aggop[0].shape[-1]):

#         nz = grid.list_cut_aggs[i]

#         learnables = grid.learn_nodes[i]

        
#         L_i[i] = torch.zeros(len(nz),len(nz)).double().to(device)

#         list_idx = []

#         for l in learnables:
#             list_idx.append(nz.index(l))
        
#         L_i[i][np.ix_(list_idx, list_idx)] = L[np.ix_(learnables, learnables)]
        

#     return L_i
def get_Li (L, grid, train):
    
    # if train:
        
    #     L_i = {}

    #     for i in range(grid.aggop[0].shape[-1]):

    #         nz = grid.list_cut_aggs[i]

    #         learnables = grid.learn_nodes[i]

            
    #         L_i[i] = torch.zeros(len(nz),len(nz)).double().to(device)

    #         list_idx = []

    #         for l in learnables:
    #             list_idx.append(nz.index(l))
            
    #         L_i[i][np.ix_(list_idx, list_idx)] = L[np.ix_(learnables, learnables)]
            

    #     return L_i
    
    # else:
        
        L_i = {}
    
        for i in range(grid.aggop[0].shape[-1]):
    
            nz = grid.list_cut_aggs[i]
    
            learnables = grid.learn_nodes[i]
            
            idx_r = torch.tensor([learnables, np.arange(len(learnables)).tolist()]).to(device)
            r = spml.SparseCSRTensor(torch.sparse_coo_tensor(idx_r, torch.ones(len(learnables)),(L.shape[0], len(learnables))).to(device).double())#.to_sparse_csr()
            
            L_i[i] = torch.zeros(len(nz),len(nz)).double().to(device)
    
            list_idx = []
            
            for l in learnables:
                list_idx.append(nz.index(l))
            
            idx_s = torch.tensor([list_idx, np.arange(len(list_idx)).tolist()]).to(device)
            s = spml.SparseCSRTensor(torch.sparse_coo_tensor(idx_s, torch.ones(len(list_idx)),(len(nz), len(list_idx))).to(device).double())#.to_sparse_csr()
    
            # L_i[i][np.ix_(list_idx, list_idx)] = L[np.ix_(learnables, learnables)]
    
            L_i[i] = s @ (r.T @ L @ r) @ s.T#t()
            
    
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
        A = scipy.sparse.coo_matrix(A)
        idxs = torch.tensor(np.array(A.nonzero()))
        dat = torch.tensor(A.data)
    else:
        idxs = torch.tensor([[i//A.shape[1] for i in range(A.shape[0]*A.shape[1])], 
                             [i% A.shape[1] for i in range(A.shape[0]*A.shape[1])]])
        dat = A.flatten()
    s = torch.sparse_coo_tensor(idxs, dat, (A.shape[0], A.shape[1]))
    return s#.to_sparse_csr()

    
def preconditioner(grid, output, train = False, precond_type = False, u = None):
    
    
    output, out_R0 = output
    
    M = 0
    
    if train:
        # tsA = torch.tensor(grid.A.toarray()).to(device)
        tsA = spml.SparseCSRTensor(make_sparse_torch(grid.A)).to(device)
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
                A_inv = torch.linalg.pinv((make_sparse_torch(grid.R_hop[i]).to_sparse_csr() @ make_sparse_torch(grid.A).to_sparse_csr() @ make_sparse_torch(grid.R_hop[i]).to_sparse_csr().t()).to_dense())
    
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
            L = get_Li (masked, grid, train)
    
            for i in range(grid.aggop[0].shape[-1]):
                
    
                r0 = grid.R[i].nonzero()[-1].tolist()
                rdelta = grid.R_hop[i].nonzero()[-1].tolist()
                list_ixs = []
                for e in r0:
                    list_ixs.append(rdelta.index(e))
                    
                modified_R_i = grid.modified_R[i]
                
    
                modified_L = L[i].to(device)
                grid_Rhop_i = spml.SparseCSRTensor(make_sparse_torch(grid.R_hop[i])).to(device)
                # grid_Rhop_i = make_sparse_torch(grid.R_hop[i]).to_dense().to(device)
    
                AA =  grid_Rhop_i @ tsA @ grid_Rhop_i.T  ####SPSPMM
    
                # AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
                A_tilde_inv = torch.linalg.pinv(AA.to_dense() + (1/(grid.h**2))*modified_L.to_dense())
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
    else:
        tsA = make_sparse_torch(grid.A).to_sparse_csr().to(device)

        if precond_type == 'RAS':
    
            for i in range(grid.aggop[0].shape[-1]):
                
                sp_R_hop_i = make_sparse_torch(grid.R_hop[i]).to_sparse_csr().to(device)
                # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
                A_inv = torch.linalg.pinv((sp_R_hop_i @ tsA @ sp_R_hop_i.t()).to_dense())
    
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
                indices_mat = torch.tensor([row_idx, col_idx]).to(device)
                add_M = torch.sparse_coo_tensor(indices_mat, A_inv[list_ixs, :].flatten(), (grid.A.shape[0], grid.A.shape[1])).to(device)
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
            L = get_Li (masked, grid, train)
    
            for i in range(grid.aggop[0].shape[-1]):
                
    
                r0 = grid.R[i].nonzero()[-1].tolist()
                rdelta = grid.R_hop[i].nonzero()[-1].tolist()
                list_ixs = []
                for e in r0:
                    list_ixs.append(rdelta.index(e))
                    
                modified_R_i = grid.modified_R[i]
                
    
                modified_L = L[i].to(device)
                grid_Rhop_i = make_sparse_torch(grid.R_hop[i]).to_sparse_csr().to(device)
                # grid_Rhop_i = make_sparse_torch(grid.R_hop[i]).to_dense().to(device)
    
                AA =  grid_Rhop_i @ tsA @ grid_Rhop_i.t()  ####SPSPMM
    
                # AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
                A_tilde_inv = torch.linalg.pinv(AA.to_dense() + (1/(grid.h**2))*modified_L.to_dense())
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
    
    return M
        
        

    
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



def stationary_cycle(A, M, R0, err, train):

    
    if type(A) == torch.Tensor:
        R0_transpose = R0.t().to(device)
        A0 = R0 @ A @ R0_transpose
        if train:
            A0_inv = torch.linalg.pinv(A0).to(device)
        else:
            A0_inv = torch.linalg.pinv(A0.to_dense()).to(device)

    if type(A) == scipy.sparse.csr.csr_matrix:
        
        R0_transpose = R0.transpose()#.to(device)
        A0 = R0 @ A @ R0_transpose
        A0_inv = np.linalg.pinv(A0.toarray())#.to(device)
        
    if type(A) == numml.sparse.SparseCSRTensor:

        R0_transpose = R0.T.to(device)
        A0 = R0 @ A @ R0_transpose
        A0_inv = torch.linalg.pinv(A0.to_dense()).to(device)
        # A0_inv = spml.SparseCSRTensor(A0_inv).to(device)

    # e = err
    # e = e - M @ (A @ e)
    # e = e - R0_transpose @ (A0_inv @ (R0 @ (A @ e)))
    
    e = err
    e = A @ e
    e = M @ e
    e = err - e
    err_1 = e
    e = A @ e
    e = R0 @ e
    e = A0_inv @ e
    e = R0_transpose @ e
    e = err_1 - e
    
    return e
    
    
def stat_cyc(A, M, R0, err, train, level):

    
    if type(A) == torch.Tensor:
        R0_transpose = R0.t().to(device)
        A0 = R0 @ A @ R0_transpose
        if train:
            A0_inv = torch.linalg.pinv(A0).to(device)
        else:
            A0_inv = torch.linalg.pinv(A0.to_dense()).to(device)

    if type(A) == scipy.sparse.csr.csr_matrix:
        
        R0_transpose = R0.transpose()#.to(device)
        A0 = R0 @ A @ R0_transpose
        A0_inv = np.linalg.pinv(A0.toarray())#.to(device)
        
    if type(A) == numml.sparse.SparseCSRTensor:
        
        R0_transpose = R0.T.to(device)
        A0 = R0 @ A @ R0_transpose
        A0_inv = torch.linalg.pinv(A0.to_dense()).to(device)
        # A0_inv = spml.SparseCSRTensor(A0_inv).to(device)

    # e = err
    # e = e - M @ (A @ e)
    # e = e - R0_transpose @ (A0_inv @ (R0 @ (A @ e)))
    if level == 'fine':
        
        e = err
        e = A @ e
        e = M @ e
        e = err - e
        
    if level == 'coarse':
        e = err
        err_1 = e
        e = A @ e
        e = R0 @ e
        e = A0_inv @ e
        e = R0_transpose @ e
        e = err_1 - e
    
    return e


def stationary_max(grid, out, u = None, K = None, precond_type = 'ML_ORAS'):
    # t0 = time.time()

    M = preconditioner(grid, out, train = True, precond_type = precond_type, u = u)#.to_dense()

    # t1 = time.time()
    # return torch.norm(eprop)
    
    list_l2 = []
    
    
    # out_lmax = copy.deepcopy(u)#spml.SparseCSRTensor(copy.deepcopy(u))
    # list_max = torch.zeros(K).to(device)
    # tsA = make_sparse_torch(grid.A).to_dense().to(device)#spml.SparseCSRTensor(make_sparse_torch(grid.A))
    # R0 = out[1].to_dense()#spml.SparseCSRTensor(out[1])
    
    list_max = torch.zeros(K).to(device)
    out_lmax = copy.deepcopy(u).float()#spml.SparseCSRTensor(copy.deepcopy(u))
    tsA = spml.SparseCSRTensor(make_sparse_torch(grid.A))
    R0 = out[1]#spml.SparseCSRTensor(out[1])
    M = spml.SparseCSRTensor(M)

    for k in range(K):
        # out_lmax = eprop @ out_lmax

        out_lmax = stationary_cycle(tsA, M, R0, out_lmax, train = True) #+ out_lmax*1e-2

        l2 = torch.norm(out_lmax, p='fro', dim = 0)

        list_max[k] = max(l2) ** (1/(k+1))
        list_l2.append(l2)
        
        # out_lmax = out_lmax/(((out_lmax**2).sum(0))**0.5).unsqueeze(0)
    # t2 = time.time()
    

    # L_max = max(list_l2[-1])#torch.dot(softmax(conv_fact), conv_fact)
    L_max = (torch.softmax(list_max, dim = 0) * list_max).sum()#max(list_max)

    # print(f'The Normalized List = {np.round(np.array(list_max.detach()),2)} \n')
    # print(f'The actual List = {np.round(np.array(list_max_2.detach()),2)} \n')

    # print('****************')
    
    # mloras_Pcol_norm = 0
    # ras_Pcol_norm = 0
    
    # for i in range(grid.R0.shape[0]):
        
    #     mloras_Pcol_norm += out[1][i] @ tsA @ out[1].T()[:,i]
    #     r0 = torch.tensor(grid.R0.toarray())
    #     ras_Pcol_norm += r0[i] @ tsA @ r0.t()[:,i]
    
    # pcol_loss = mloras_Pcol_norm/ras_Pcol_norm
    
    return L_max #+ pcol_loss*5.0



def torch_2_scipy_sparse(A):
    
    data = A.coalesce().values()
    row = A.coalesce().indices()[0]
    col = A.coalesce().indices()[1]
    out = scipy.sparse.csr_matrix((data, (row, col)), shape=(A.shape[0], A.shape[1]))
    
    return out


def test_stationary(grid, output, precond_type, u, K, M):

    
    out = copy.deepcopy(u).float().to(device)
    l2_list = []
    vec_list = []
    l2 = torch.norm(out, p='fro', dim = 0) #np.linalg.norm(out, axis = 0)
    l2_list.append(max(l2))
    vec_list.append(out[:,np.argmax(l2)])
    tsA = spml.SparseCSRTensor(make_sparse_torch(grid.A).float())
    M = spml.SparseCSRTensor(M.float())
    
    if precond_type == 'ML_ORAS':

        R0 = output[1].to(device)

    else:
        R0 = spml.SparseCSRTensor(make_sparse_torch(grid.R0).float()).to(device)

    for k in range(K):
        # out = eprop_a @ out
        out = stationary_cycle(tsA, M, R0, out,train = False)
        
        l2 = torch.norm(out, p='fro', dim = 0)
        l2_list.append(max(l2))
        vec_list.append(out[:,np.argmax(l2)])


    return l2_list, vec_list

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



