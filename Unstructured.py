import numpy as np
import sys
# sys.path.append('utils')
# import matplotlib.pyplot as plt
import scipy
import fem
import pygmsh
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import random
import torch as T
import torch_geometric
import copy
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
from pyamg.gallery import poisson
from torch_geometric.data import Data
from pyamg.aggregation import lloyd_aggregation
# import matplotlib as mpl
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr, isspmatrix_csc
from pyamg.graph import lloyd_cluster
# from matplotlib.pyplot import figure, text
import torch_geometric.data

from torch_geometric.utils.num_nodes import maybe_num_nodes

# mpl.rcParams['figure.dpi'] = 300


class MyMesh:
    def __init__(self, mesh):
        
        self.nv = mesh.points[:,0:2].shape[0]
        self.X = mesh.points[:,0:1].flatten() * ((self.nv/50)**0.5)
        self.Y = mesh.points[:,1:2].flatten() * ((self.nv/50)**0.5)

        self.E = mesh.cells[1].data
        self.V = mesh.points[:,0:2]
        
        self.ne = len(mesh.cells[1].data)
        
        e01 = self.E[:,[0,1]]
        e02 = self.E[:,[0,2]]
        e12 = self.E[:,[1,2]]
    
        e01 = tuple(map(tuple, e01))
        e02 = tuple(map(tuple, e02))
        e12 = tuple(map(tuple, e12))
        
        e = list(set(e01).union(set(e02)).union(set(e12)))
        self.N = [i for i in range(self.X.shape[0])]
        self.Edges = e
        self.num_edges = len(e)
        
      
def structured(n_row, n_col, Theta):
    
    num_nodes = int(n_row*n_col)

    X = np.array([[i*0.04 for i in range(n_col)] for j in range(n_row)]).flatten()
    Y = np.array([[j*0.04 for i in range(n_col)] for j in range(n_row)]).flatten()
    E = []
    V = []
    nv = num_nodes
    N = [i for i in range(num_nodes)]
    
    epsilon = 1
    theta = 1 #param of A matrix
   
    sten = diffusion_stencil_2d(epsilon=epsilon,theta=theta,type='FD')
    AA = stencil_grid(sten, (n_row, n_col), dtype=float, format='csr')
    
    A = AA.toarray()
    
    nz_row = np.nonzero(A)[0]
    nz_col = np.nonzero(A)[1]
    e = np.concatenate((np.expand_dims(nz_row,axis=1), np.expand_dims(nz_col, axis=1)), axis=1)
    Edges = list(tuple(map(tuple, e)))
    num_edges = len(Edges)
    g = rand_grid_gen(None)
    
    mesh = copy.deepcopy(g.mesh)

    mesh.X = X
    mesh.Y = Y
    mesh.E = []
    mesh.V = V
    mesh.nv = nv
    mesh.ne = []
    mesh.N = N
    mesh.Edges = Edges
    mesh.num_edges = num_edges
    
    
    Neumann = True
    
    if Neumann:
        
        boundary_3 = []
        for i in range(n_row):
            
            if i == 0 or i == n_row-1:
                boundary_3.extend([i*n_col + j for j in range(n_col)])
            else:
                boundary_3.extend([i*n_col, i*n_col+n_col-1])
            
        boundary_2 = [0, n_col-1, (n_row-1)*n_col, n_row*n_col-1]
        
        for i in boundary_3:
            
            AA[i,i] = 3.0
        
        for i in boundary_2:
            
            AA[i,i] = 2.0
        
    fine_nodes = [i for i in range(num_nodes)]
    
    grid_ = Grid(AA,fine_nodes,[], mesh, Theta)
    
    return grid_
     

def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """
    import networkx as nx

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def from_scipy_sparse_matrix(A):
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    A = A.tocoo()
    row = T.from_numpy(A.row).to(T.long)
    col = T.from_numpy(A.col).to(T.long)
    edge_index = T.stack([row, col], dim=0)
    edge_weight = T.from_numpy(A.data)
    return edge_index, edge_weight


def to_scipy_sparse_matrix(edge_index, edge_attr=None, num_nodes=None):
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    row, col = edge_index.cpu()

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1).cpu()
        assert edge_attr.size(0) == row.size(0)

    N = maybe_num_nodes(edge_index, num_nodes)
    out = scipy.sparse.coo_matrix(
        (edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
    return out


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = T.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = T.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data



class Grid(object):
    
    def __init__(self, A, mesh):

        self.A = A.tocsr()

        self.num_nodes = mesh.nv
        #self.edges = set_edge
        self.mesh = mesh
        
        active = np.ones(self.num_nodes)
        self.active = active
        
        self.G = nx.from_scipy_sparse_matrix(self.A, edge_attribute='weight', parallel_edges=False)

  
        self.x = T.cat((T.from_numpy(self.active).unsqueeze(1), \
                        T.from_numpy(self.active).unsqueeze(1)),dim=1).float()

        
        edge_index, edge_attr = from_scipy_sparse_matrix(abs(self.A))
        edge_index4P, edge_attr4P = from_scipy_sparse_matrix(self.A)
        
        list_neighbours1 = []
        list_neighbours2 = []
        for node in range(self.num_nodes):
            a =  list(self.G.edges(node,data = True))
            l1 = []
            l2 = []
            for i in range(len(a)):
                l1.append(a[i][1])
                l2.append(abs(np.array(list(a[i][-1].values())))[0])
                
            list_neighbours1.append(l1)
            list_neighbours2.append(l2)
                
        self.list_neighbours = [list_neighbours1, list_neighbours2]
        
        self.data = Data(x=self.x, edge_index=edge_index, edge_attr= edge_attr.float())
        self.data4P = Data(x=self.x, edge_index=edge_index4P, edge_attr= edge_attr4P.float())

        
    def subgrid(self, node_list):

        sub_x = self.x[node_list]
        sub_data = from_networkx(self.G.subgraph(node_list))
        sub_data = Data(x=sub_x, edge_index=sub_data.edge_index, edge_attr= abs(sub_data.weight.float()))
        
        return sub_data
    
        
    def node_hop_neigh(self, node, K):
        
        return list(nx.single_source_shortest_path(self.G, node, cutoff=K).keys())
    
    
    def plot(self, size, w, labeling, fsize):
        
        G = nx.from_scipy_sparse_matrix(self.A)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        mymsh = self.mesh
        
        # points = mymsh.N
        # edges  = mymsh.Edges
        
        pos_dict = {}
        for i in range(mymsh.nv):
            pos_dict[i] = [mymsh.X[i], mymsh.Y[i]]
            
        # G.add_nodes_from(points)
        # G.add_edges_from(edges)
        colors = [i for i in range(mymsh.nv)]
        
        for i in range(self.num_nodes):
            colors[i] = 'r'

        
        draw_networkx(G, pos=pos_dict, with_labels=labeling, node_size=size, \
                      node_color = colors, node_shape = 'o', width = w, font_size = fsize)
        
        plt.axis('equal')



def grid_subdata_and_plot(node, cutoff, grid_, ploting = False, labeling = True, size = 300.0, w = 1.0):
    
    node_list = list(nx.single_source_dijkstra_path_length(grid_.G, 
                                            node, cutoff = cutoff, weight=None).keys())
    act_coarse_list = []
    sub_x = grid_.data.x[node_list][:,0]
    sub_data = from_networkx(grid_.G.subgraph(node_list))
    sub_data = Data(x=sub_x, edge_index=sub_data.edge_index,
                    edge_attr= abs(sub_data.weight.float()))
    
    G = grid_.G.subgraph(node_list)

    mymsh = grid_.mesh
    node_list = list(G.nodes)
    sub_data.x = grid_.data.x[node_list][:,0]
    
    if ploting:
        
        pos_dict = {}
        for i in node_list:
            pos_dict[i] = [mymsh.X[i], mymsh.Y[i]]
    
        colors = [i for i in node_list]
        
        
        for i in range(len(node_list)):
            if node_list[i] in list(set(grid_.fine_nodes) - set(grid_.coarse_nodes)):
                colors[i] = 'b'
            if node_list[i] in grid_.coarse_nodes:
                act_coarse_list.append(node_list[i])
                colors[i] = 'r'
            # if node_list[i] in grid_.viol_nodes()[0]:
            #     colors[i] = 'g'
    
        
        draw_networkx(G, pos=pos_dict, with_labels=labeling, node_size=size
                      , \
                      node_color = colors, node_shape = 'o', width = w, font_size=5)
    
            
        plt.axis('equal')
    
    idx_dict = {}
    for i in range(len(node_list)):
        idx_dict[node_list[i]] = i
    
    after_coarse_list = np.nonzero(sub_data.x == 0).flatten().tolist()
    
    spmtrx = to_scipy_sparse_matrix(sub_data.edge_index, edge_attr=sub_data.edge_attr)
    GG = nx.from_scipy_sparse_matrix(spmtrx, edge_attribute='weight', parallel_edges=False)
    
    return sub_data, node_list, act_coarse_list, after_coarse_list, idx_dict, GG
    

def lloyd_aggregation(C, ratio=0.03, distance='unit', maxiter=10000):
    """Aggregate nodes using Lloyd Clustering.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix
    ratio : scalar
        Fraction of the nodes which will be seeds.
    distance : ['unit','abs','inv',None]
        Distance assigned to each edge of the graph G used in Lloyd clustering

        For each nonzero value C[i,j]:

        =======  ===========================
        'unit'   G[i,j] = 1
        'abs'    G[i,j] = abs(C[i,j])
        'inv'    G[i,j] = 1.0/abs(C[i,j])
        'same'   G[i,j] = C[i,j]
        'sub'    G[i,j] = C[i,j] - min(C)
        =======  ===========================

    maxiter : int
        Maximum number of iterations to perform

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    seeds : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import lloyd_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> lloyd_aggregation(A)[0].todense() # one aggregate
    matrix([[1],
            [1],
            [1],
            [1]], dtype=int8)
    >>> # more seeding for two aggregates
    >>> Agg = lloyd_aggregation(A,ratio=0.5)[0].todense()

    """
    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    if not (isspmatrix_csr(C) or isspmatrix_csc(C)):
        raise TypeError('expected csr_matrix or csc_matrix')

    if distance == 'unit':
        data = np.ones_like(C.data).astype(float)
    elif distance == 'abs':
        data = abs(C.data)
    elif distance == 'inv':
        data = 1.0/abs(C.data)
    elif distance is 'same':
        data = C.data
    elif distance is 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError('unrecognized value distance=%s' % distance)

    if C.dtype == complex:
        data = np.real(data)

    assert(data.min() >= 0)

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    num_seeds = int(min(max(ratio * G.shape[0], 1), G.shape[0]))

    distances, clusters, seeds = lloyd_cluster(G, num_seeds, maxiter=maxiter)

    row = (clusters >= 0).nonzero()[0]
    col = clusters[row]
    data = np.ones(len(row), dtype='int8')
    AggOp = coo_matrix((data, (row, col)),
                       shape=(G.shape[0], num_seeds)).tocsr()
    
    return AggOp, seeds, col


        
def set_edge_from_msh(msh):
        
    edges = msh.E
    array_of_tuples = map(tuple, edges[:,[1,2]])
    t12 = tuple(array_of_tuples)
    array_of_tuples = map(tuple, edges[:,[0,2]])
    t02 = tuple(array_of_tuples)
    array_of_tuples = map(tuple, edges[:,[0,1]])
    t01 = tuple(array_of_tuples)
    
    set_edge = set(t01).union(set(t02)).union(set(t12))
    
    return set_edge



def rand_Amesh_gen3(kappa = None, gamma = None, PDE='Poisson'):
    
    num_Qhull_nodes = random.randint(45, 90)
    points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
    hull = ConvexHull(points)
    
    msh_sz = 0.15*random.random()+0.25
    mesh = None
    if mesh == None:
    
        
        with pygmsh.geo.Geometry() as geom:
            
            poly = geom.add_polygon(
                
                    hull.points[hull.vertices.tolist()].tolist()
                    
                ,
                mesh_size=msh_sz,
            )
            mesh = geom.generate_mesh()
        
        '''
            geom.set_mesh_size_callback(
                lambda dim, tag, x, y, z: 6.0e-2 + 2.0e-1 * (x ** 2 + y ** 2)
            )
            
            field0 = geom.add_boundary_layer(
                edges_list=[poly.curves[0], poly.curves[2]],
                lcmin=0.05,
                lcmax=0.2,
                distmin=0.0,
                distmax=0.2,
            )
            field1 = geom.add_boundary_layer(
                nodes_list=[poly.points[8], poly.points[2]],
                lcmin=0.05,
                lcmax=0.2,
                distmin=0.1,
                distmax=0.4,
            )
            geom.set_background_mesh([field0, field1], operator="Min")
      
            mesh = geom.generate_mesh()
        
        
        '''
        
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(
                [
                    [-1.0, -1.0],
                    [+1.0, -1.0],
                    [+1.0, +1.0],
                    [-1.0, +1.0],
                ]
            )
            geom.set_mesh_size_callback(
                lambda dim, tag, x, y, z: 9.0e-2 + 3.0e-1 * (x ** 2 + y ** 2)
            )
        
            mesh = geom.generate_mesh()
        
        '''
        
        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(
                [
                    
                    
                    [0.0, 0.0],
                    [0.7, 0.5],
                    [1.0, 0.0],
                    
                    
                    
                ],
                mesh_size=1
                ,
            )
        
            field0 = geom.add_boundary_layer(
                edges_list=[poly.curves[0], poly.curves[1], poly.curves[2]],
                lcmin=0.01,
                lcmax=0.1,
                distmin=0.15,
                distmax=0.01,
            )
            field1 = geom.add_boundary_layer(
                nodes_list=[poly.points[2]],
                lcmin=0.05,
                lcmax=0.2,
                distmin=0.1,
                distmax=0.4,
            )
            geom.set_background_mesh([field0], operator="Min")
        
            mesh = geom.generate_mesh()
          
         '''
            
    mymsh = MyMesh(mesh)
    # points = mymsh.V
    # tri = Delaunay(points)
    # plt.triplot(points[:,0], points[:,1], tri.simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    
    A,b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1, gamma=gamma , PDE=PDE)
    return A, mymsh

        

def func2(x,y,p):
    
    x_f = int(np.floor(p.shape[0]*x))
    y_f = int(np.floor(p.shape[1]*y))
    
    return p[x_f, y_f]

def rand_Amesh_gen2(randomized, n, var = 0.01,  min_ = 0.01, min_sz = 0.1, kappa=None, gamma=None, PDE='Poisson'):
    
    num_Qhull_nodes = random.randint(3, 45)
    if randomized:
        points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
    
    else:
        points = []
    
        for i in range(1,n+1):
            points.append([0.5+0.2*np.cos(i*2*np.pi/n + np.pi/n), 0.5+0.2*np.sin(i*2*np.pi/n + np.pi/n)])
        hull = ConvexHull(points)
        points = np.array(points)
        
    msh_sz = 0.01 #0.1*random.random()+0.1
    
    
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            
                hull.points[hull.vertices.tolist()].tolist()
                
            ,
            mesh_size=msh_sz,
        )

            
        min_c = min_ + var*np.random.random()

        p = min_c + min_sz*np.random.random((500,500))
        geom.set_mesh_size_callback(
            #lambda dim, tag, x, y, z: func(x, y, points,min_dist, thresh, min_sz)
            lambda dim, tag, x, y, z: func1(x, y, p)
        )
        
        #geom.set_background_mesh([field0, field1], operator="Min")
        
        mesh = geom.generate_mesh()
    
    
    
    mymsh = MyMesh(mesh)
    
    A,b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1, gamma=gamma , PDE=PDE)
    
    return A, mymsh


def func1(x,y,p):
    
    x_f = int(np.floor(p.shape[0]*x))
    y_f = int(np.floor(p.shape[1]*y))
    
    return p[x_f, y_f]

def rand_Amesh_gen1(randomized, n, min_, min_sz, lcmin, lcmax, distmin, distmax, kappa=None, gamma=None, PDE='Poisson', kap=1.0):
    
    num_Qhull_nodes = random.randint(3, 45)
    if randomized:
        points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
        hull = ConvexHull(points)
    else:
        points = []
    
        for i in range(1,n+1):
            points.append([0.5+0.2*np.cos(i*2*np.pi/n + np.pi/n), 0.5+0.2*np.sin(i*2*np.pi/n + np.pi/n)])
        hull = ConvexHull(points)
        points = np.array(points)
    
    msh_sz = 0.1 #0.1*random.random()+0.1
    
        
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            
                hull.points[hull.vertices.tolist()].tolist()
                
            ,
            mesh_size=msh_sz,
        )
        
        p = 0.6 + 0.5*np.random.random((500,500))
        geom.set_mesh_size_callback(
            #lambda dim, tag, x, y, z: func(x, y, points,min_dist, thresh, min_sz)
            lambda dim, tag, x, y, z: func1(x, y, p)
        )
        
        n_edge = len(poly.curves)
        list_edge_idx = np.random.randint(0, n_edge, np.random.randint(1,3,1).item())
        edges_list = [poly.curves [i] for i in list_edge_idx]
        
        n_points = len(poly.points)
        list_point_idx = np.random.randint(0, n_points, np.random.randint(1,5,1).item())
        nodes_list = [poly.points [i] for i in list_point_idx]
        
        field0 = geom.add_boundary_layer(
            edges_list=edges_list,
            lcmin=lcmin,
            lcmax=lcmax,
            distmin=distmin,
            distmax=distmax,
        )
        field1 = geom.add_boundary_layer(
            nodes_list=nodes_list,
            lcmin=lcmin,
            lcmax=lcmax,
            distmin=distmin,
            distmax=distmax,
        )
        geom.set_background_mesh([field0, field1], operator="Min")
  
        mesh = geom.generate_mesh()
        
    mymsh = MyMesh(mesh)
    # points = mymsh.V
    # tri = Delaunay(points)
    # plt.triplot(points[:,0], points[:,1], tri.simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    
    A,b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1, gamma=gamma , PDE=PDE, kap=kap)
    return A, mymsh



def rand_Amesh_gen(mesh_size, kappa = None, gamma = None, PDE='Poisson'):
    
    num_Qhull_nodes = random.randint(10,45)
    points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
    hull = ConvexHull(points)
    
    msh_sz = mesh_size
    
    
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            
                hull.points[hull.vertices.tolist()].tolist()
                
            ,
            mesh_size=msh_sz,
        )
        
        # prob = np.random.random()
        # if prob>5:
            
        #     min_ = 0.005+0.01*np.random.random()
        #     min_sz  = 0.1#/(min_**0.1)
        #     p = min_ + min_sz*np.random.random((500,500))
        #     geom.set_mesh_size_callback(
        #         #lambda dim, tag, x, y, z: func(x, y, points,min_dist, thresh, min_sz)
        #         lambda dim, tag, x, y, z: func1(x, y, p)
        #     )
            
        #     #geom.set_background_mesh([field0, field1], operator="Min")
        
        mesh = geom.generate_mesh()
    
    
    
    mymsh = MyMesh(mesh)
    # factor = (mymsh.nv/100) ** 0.5
    # mymsh.V = mymsh.V * factor
    # mymsh.X = mymsh.X * factor
    # mymsh.Y = mymsh.Y * factor
    
    # points = mymsh.V
    # tri = Delaunay(points)
    # plt.triplot(points[:,0], points[:,1], tri.simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    
    A,b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1, gamma=gamma , PDE=PDE)
    
    return A, mymsh


#T.save(mesh, "mesh.pth")
#mesh = T.load("mesh.pth")

def rand_grid_gen3(kappa = None, gamma = None, PDE='Poisson'):
    
    A, mymsh = rand_Amesh_gen3(kappa = kappa, gamma = gamma, PDE=PDE)
    
    rand_grid = Grid(A,mymsh)
    
    return rand_grid

def rand_grid_gen2(randomized, n, var = 0.01, min_ = 0.01, min_sz = 0.1, kappa = None, gamma = None, PDE='Poisson'):
    
    A, mymsh = rand_Amesh_gen2(randomized, n, var, min_, min_sz, kappa, gamma, PDE)
    
    rand_grid = Grid(A,mymsh)
    
    return rand_grid
def rand_grid_gen1(randomized, n, min_, min_sz, lcmin, lcmax,distmin, distmax, kappa = None, gamma = None, PDE='Poisson', kap=1.0):
    
    A, mymsh = rand_Amesh_gen1(randomized, n, min_, min_sz, lcmin, lcmax, distmin, 
                               distmax, kappa = kappa, gamma = gamma, PDE=PDE, kap=kap)
    
    rand_grid = Grid(A,mymsh)
    
    return rand_grid

def rand_grid_gen(mesh_sz, kappa = None, gamma = None, PDE='Poisson'):
    
    A, mymsh = rand_Amesh_gen(mesh_sz, kappa = kappa, gamma = gamma, PDE = PDE)
    
    rand_grid = Grid(A,mymsh)
    
    return rand_grid



import gmsh
import torch
import meshio
#mesh = meshio.read('Test_Graphs/Hand_crafted/Geometry/Graph1.msh')

class gmsh2MyMesh:
    def __init__(self, mesh):
        
        
        diff = set([i for i in range(mesh.points[:,0:2].shape[0])]) - \
            set(mesh.cells[-1].data.flatten().tolist())
            
        mesh.points = np.delete(mesh.points, list(diff), axis=0)
        arr_diff = np.array(list(diff))
        for i in range(len(mesh.cells[-1].data)):
            
            for j in range(3):
                
                shift = mesh.cells[-1].data[i,j]>arr_diff
                shift = np.sum(shift)
                mesh.cells[-1].data[i,j] = mesh.cells[-1].data[i,j] - shift
            
        self.X = mesh.points[:,0:1].flatten()
        self.Y = mesh.points[:,1:2].flatten()
        self.E = mesh.cells[-1].data
        self.V = mesh.points[:,0:2]
        self.nv = mesh.points[:,0:2].shape[0]
        self.ne = len(mesh.cells[-1].data)
        
        e01 = self.E[:,[0,1]]
        e02 = self.E[:,[0,2]]
        e12 = self.E[:,[1,2]]
    
        e01 = tuple(map(tuple, e01))
        e02 = tuple(map(tuple, e02))
        e12 = tuple(map(tuple, e12))
        
        e = list(set(e01).union(set(e02)).union(set(e12)))
        self.N = [i for i in range(self.X.shape[0])]
        self.Edges = e
        self.num_edges = len(e)
    

def hand_grid(mesh):
    
    msh = gmsh2MyMesh(mesh)
    
    A,b = fem.gradgradform(msh, kappa=None, f=None, degree=1)
    
    fine_nodes = [i for i in range(A.shape[0])]
    
    #set_of_edge = set_edge_from_msh(mymsh)
    rand_grid = Grid(A,fine_nodes,[],msh,0.56)
    
    return rand_grid



def data2interpole(G, num_nodes, coarse_nodes):
    
    g = nx.Graph(G)
    fine_nodes = list(set([i for i in range(num_nodes)])-set(coarse_nodes))
    Hc = g.subgraph(coarse_nodes)
    coarse2_remove_edge = Hc.edges
    g.remove_edges_from(coarse2_remove_edge)
    
    Hf = g.subgraph(fine_nodes)
    fine2_remove_edge = Hf.edges
    g.remove_edges_from(fine2_remove_edge)
    data = from_networkx(g)
    
    edge_attr  = abs(data.weight)
    edge_index = data.edge_index
    x          = T.zeros(num_nodes)
    x[coarse_nodes] = 1.0
    x = x.unsqueeze(1).float()
    output = Data(x=x, edge_index=edge_index, edge_attr= edge_attr.float())
    
    return output
    

    
