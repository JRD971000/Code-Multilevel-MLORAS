import numpy as np
import scipy
import torch
from grids import *
import matplotlib.pyplot as plt


def struct_agg(n_row, n_col, agg_row, agg_col):
    n_row -= 1
    n_col -= 1
    arg0 = 0
    arg2 = []
    d = int(n_col/agg_col)
    for i in range(n_row * n_col):
        j = i%n_col
        k = i//n_col
        arg2.append(int(j//agg_col) + (k//agg_row)*d)
        
    arg1 = np.zeros(arg2[-1]+1)

    return (arg0, arg1, arg2)

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


def get_overlaps(grid):
    
    num_nodes = grid.A.shape[0]
    dim = int(num_nodes ** 0.5)
    l = grid.R_hop[0].shape[0]
    overlap0 = [-i - 1 + l for i in range(dim)]
    overlap1 = [num_nodes - l + i for i in range(dim)]
    
    return [overlap0, overlap1]

def classical_RAS(grid):
    
    M = 0
    
    for i in range(grid.aggop[0].shape[-1]):
        
        A_inv = torch.linalg.pinv(torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray()))
        # A_inv = torch.tensor(torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray()))
    
        ####
        
        r0 = grid.R[i].toarray().nonzero()[-1].tolist()
        rdelta = grid.R_hop[i].toarray().nonzero()[-1].tolist()
        list_ixs = []
        for e in r0:
            list_ixs.append(rdelta.index(e))
            
        modified_R_i = np.zeros_like(torch.tensor(grid.R_hop[i].toarray()))
        modified_R_i[list_ixs, :] = grid.R[i].toarray()
        ####
        M += torch.tensor(modified_R_i.transpose()) @ A_inv @ torch.tensor(grid.R_hop[i].toarray())
    
    return M

def construct_precond(grid, p, q):


    M = 0
    
    overlap = get_overlaps(grid)
    size = len(overlap[0])

    I = np.eye(size)
    T0 = scipy.sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(size, size)).toarray()
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
            
            
        ####
        
        # A_tilde_inv = torch.zeros(len(nz_delta), len(nz_delta)k).double()
        # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
        # A_tilde_inv[np.ix_(nonz, nonz)] = A_inv
        
        
        AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
        AA[np.ix_(list_idx, list_idx)] = torch.tensor(T_tilde)/(grid.h ** 2)
        A_tilde_inv = torch.linalg.pinv(AA)
        M += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())

    return M

def OS(grid):
    
    M = 0
    
    overlap = get_overlaps(grid)
    size = len(overlap[0])
    B = scipy.sparse.diags([-1,2, -1], [-1,0,1], shape=(size, size))/(grid.h ** 2)
    evals, evecs = np.linalg.eig(B.toarray())
    modified_evals = np.sqrt(evals ** 2 + grid.nu)
    
    T_tilde = evecs @ np.diag(modified_evals) @ evecs.transpose()
    
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
            
        
        AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
        AA[np.ix_(list_idx, list_idx)] = torch.tensor(T_tilde)
        A_tilde_inv = torch.linalg.pinv(AA)
        M += torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())
        
    return M

softmax = torch.nn.Softmax(dim=0)

def test_stationary(grid, M, u, K):
    
       
    eprop_a = torch.eye(M.shape[0]) - M @ torch.tensor(grid.A.toarray())
    
    out = copy.deepcopy(u)
    l2_list = []
    l2 = torch.norm(out, p='fro', dim = 0)
    # l2_list.append(l2)
    l2_list.append(torch.dot(softmax(l2), l2))
    for k in range(K):
        out = eprop_a @ out
        l2 = torch.norm(out, p='fro', dim = 0)
        l2_list.append(torch.dot(softmax(l2), l2))
        # l2_list.append(l2)
        
    return l2_list


if __name__ == '__main__':
    
    
    n = 10
    m = int(n/2)
    cut_ = 1
    
    
    old_g = structured(n, n)#torch.load("3s_reg.pth")#
    print(old_g.num_nodes)
    grid =  Grid_PWA(old_g.A, old_g.mesh, 0.02, hops = 0, cut=cut_, h = 1/(n+1), nu = 1)
    grid.aggop_gen(ratio = 0.1, cut = cut_, node_agg = struct_agg_PWA(n,n,m,n))
    grid.plot_agg(size = 20, fsize = 3)
    plt.show()
    
    ###### p  =  np.sqrt(np.pi) * ((np.pi ** 2 + grid.nu) ** 0.25) * grid.h ** (-0.5)
    # p  = (2**(-1/3)) * ((np.pi**2 + grid.nu) ** (1/3)) * (grid.h ** (-1/3))
    # # p    = np.sqrt(grid.nu)
    # # p  =  (2+grid.nu*((grid.h)**2))/(2*grid.h)
    # # q  =  grid.h/2
    # q = 0
    p  = (2**(-3/5)) * ((np.pi**2 + grid.nu) ** (2/5)) * (grid.h ** (-1/5))
    q = (2**(-1/5)) * ((np.pi**2 + grid.nu) ** (-1/5)) * (grid.h ** (3/5))
    
    
    
    u = torch.rand(grid.x.shape[0],100).double()
    u = u/(((u**2).sum(0))**0.5).unsqueeze(0)
    K = 10
    
    M_ORAS = construct_precond(grid, p = p, q = q)
    l2_ORAS = test_stationary(grid, M_ORAS, u, K)
    
    
    M_RAS = classical_RAS(grid)
    l2_RAS = test_stationary(grid, M_RAS, u, K)
     
    
    M_OS = OS(grid)
    l2_OS = test_stationary(grid, M_OS, u, K)
    
    
    plt.plot(l2_ORAS, label = 'ORAS OO2')
    plt.plot(l2_RAS, label = 'RAS')
    plt.plot(l2_OS, label = 'Optimal Boundary (OS)')
    plt.xlabel("Iteration")
    plt.ylabel("error norm")
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    