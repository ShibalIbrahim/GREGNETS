import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg.eigen.arpack import eigsh

############################ For MKNGCNCxComb ############################
def kNN_masking(mat, K):
    """Get K-neighborhood from the graph.
    
    Args:
        mat: adjacency matrix, float numpy array of shape (num_nodes, num_nodes).
        K: top K peers for each company
    
    Returns:
        Sparsified adjacency matrix, float numpy array of shape (num_nodes, num_nodes).
    """
    co = np.copy(np.abs(mat))
    np.fill_diagonal(co, 0)
    p = co.shape[0]
    M1 = np.argsort(co,axis=1)[:,-K:]
    M = np.zeros((p,p))
    for i in range(p):
        tmp = set(M1[i,:])
        for j in range(p):
            if j == i:
                continue
            elif j in tmp:
                M[i,j] = 1
                M[j,i] = 1
    return M

############################ For GCN/NGCN ############################
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, add_selfloop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if add_selfloop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj)
    return adj_normalized.toarray()

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]).toarray())
    t_k.append(scaled_laplacian.toarray())

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(np.asarray(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian)))

    return t_k

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def create_supports(graphs, model_type='gcn', Ks=None):
    """Creates Identity and powers of adjacency matrix for each graph.
    
    Args:
      graphs: list of knowledge graphs.
      model_type: model to generate supports for, str.
        - 'gcn'.
        - 'ngcn'.
      K: powers upto which supports are generated for each graph.
        - ignored when model_type='ngcn'.
      
    Returns:
      supports: [I, G_1^1, ... G_1^K, G_2^1, ... G_2^K, ... G_R^1, ..., G_R^K].
    """
    num_graphs = len(graphs)
    supports = []
    if model_type=='gcn':
        for g in graphs:
            g_hat = preprocess_adj(g) # pre-processing from GCN paper
            supports.append(g_hat)
            
    elif model_type=='ngcn':
        supports.append(np.identity(graphs[0].shape[0]))
        for g, K in zip(graphs, Ks):
            g_hat = preprocess_adj(g) # pre-processing from GCN paper
            supports.append(g_hat)
            for i in range(K-1):
                supports.append(g_hat.dot(supports[-1]))
    else:
        raise ValueError('model_type:{} is not supported.'.format(model_type))

    return supports