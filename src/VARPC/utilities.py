from __future__ import division, print_function
import os
import sys
from contextlib import redirect_stdout
import pandas as pd
from tqdm import tnrange, tqdm_notebook
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import Math

def renormalize_cosearchfraction(cosearch = None):
    indices = np.where(np.sum(cosearch, axis=1)>0)[0]
    cosearchfrac = np.zeros(cosearch.shape,dtype=float)
    cosearchfrac[indices,:] = cosearch[indices,:]/np.sum(cosearch[indices,:], axis=1, keepdims=True)
    return cosearchfrac

def generate_sparse_cosearchfraction(cf = None, K = 50, X = None):
    TopK = cf.argsort(axis=1)[:,-1:-(1+K):-1]
#     TopK_X = np.transpose(np.array([X[:,tk] for tk in TopK]), (1, 0, 2))
    TopK_csf = np.array([cf[i,tk]/np.sum(cf[i,tk]+1e-8) for i, tk in enumerate(TopK)])
#     cf = cf.astype(np.float32)
#     cf = 0.5*(cf+cf.transpose())
    
    # Sparse
    cf_sparse = np.zeros(cf.shape)
    for i, tk in enumerate(TopK):
        cf_sparse[i,tk] = cf[i,tk]
    cf_sparse = np.array([cf_sparse[i,:]/np.sum(cf_sparse[i,:]+1e-8) for i in range(TopK.shape[0])])
    cf_sparse = cf_sparse.astype(np.float32)
    
    # Symmetric
    cf_sym = 0.5*(cf_sparse+cf_sparse.transpose())
    
    
    return cf_sym

def load_data(load_dir = None, 
              folder = None, 
              cosearch_ticker_file = None,
              residual_file = None,
              N_years = None):
    print(load_dir)
    print(residual_file)
    # Finding submatrix cosearch fraction based on columns (Tickers) in returns/volatilities file   
    df_cosearch_ticker_file = pd.read_csv(os.path.join(load_dir, folder, cosearch_ticker_file), index_col=False)    
    ticker_to_index_mapping = df_cosearch_ticker_file['Ticker'].reset_index().set_index('Ticker').to_dict()['index']

    ### Read Volatility Residuals
    df = pd.read_csv(os.path.join(load_dir, residual_file), index_col=0)
    df.index = pd.to_datetime(df.index,format="%Y-%m-%d")
    
    indices_subset = np.array([int(i) for i in np.sort(df.columns.map(ticker_to_index_mapping).dropna().values)])
    companies_subset = df_cosearch_ticker_file.loc[indices_subset,:]['Name'].values
    ticker_subset = df_cosearch_ticker_file.loc[indices_subset,:]['Ticker'].values
    ciks_subset = df_cosearch_ticker_file.loc[indices_subset,:]['CIK'].values
    
    # display(df.head())
    # ax = df[['AAL']].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=False, rot=90)

    indices = []
    companies = []
    tickers = []
    ciks = []
    for t, i, c, ck in zip(ticker_subset,indices_subset, companies_subset, ciks_subset):
        if t in df.columns:
            tickers.append(t)
            indices.append(i)
            companies.append(c)
            ciks.append(ck)
    len(indices)
    
    df = df[tickers]
    
    
    ### Search-based peers identified using cosearch from previous year
    cosearchfraction = []
    d_date_cosearch = np.timedelta64(1, 'Y')
    for y in df.index.year.unique().values: 
        date = np.datetime64('{}'.format(y))
        file = load_dir+'/'+folder+'/cosearchfraction'+'/cosearchfractiondirected{}'.format(pd.DatetimeIndex([date-d_date_cosearch]).year.values[0])+'.npz'
        cosearchfraction_t = sp.load_npz(file).toarray()
        cosearchfraction_t = cosearchfraction_t[np.ix_(indices,indices)]
        cosearchfraction_t = renormalize_cosearchfraction(cosearchfraction_t)
        cosearchfraction.append(cosearchfraction_t)
    
    years = df.index.year.unique().values
    
    df_train = df[df.index.year.isin(years[(-2-N_years):-2])]
    #df_train = df_train.resample('M').mean() #average by month
    df_val = df[df.index.year.isin(years[-2:-1])]
    #df_val = df_val.resample('M').mean() #average by month
    df_test = df[df.index.year.isin(years[-1:])]
    #df_test = df_test.resample('M').mean() #average by month 
    
    cf_train = cosearchfraction[-3]
    cf_val = cosearchfraction[-2]
    cf_test = cosearchfraction[-1]
    
    df_companies = pd.DataFrame({'Ticker': tickers,
                                 'Company': companies,
                                 'CIK': ciks
                                })

    
    return df_train, df_val, df_test, cf_train, cf_val, cf_test, df_companies


def prepare_sequences(sequences, n_steps):
    """Prepares past-times samples as sequences.
    
    Args:
        n_steps: num of past time steps to use as features/sequences, int scalar.
        sequences: 
    
    Returns:
        X: past-time samples as features, a float numpy array of shape (T, N, p).
        Y: target responses, a float numpy array of shape (T, N).
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    X = np.swapaxes(X, 1, 2)
    Y = np.array(y)
    return X, Y


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model.
    
    References:
     - ["Semi-Supervised Classification with Graph Convolutional Networks" by Thomas N. Kipf, Max Welling,
        url: https://arxiv.org/abs/1609.02907]
    
    Args:
        adj: adjacency matrix, a numpy array of shape (N, N).
    
    Returns:
        adj_normalized: normalized adjacency matrix, a numpy array of shape (N, N).
    """    
    def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) # adding identity to zero-diagonal justified for autoregression.
    return adj_normalized.toarray()

def compute_metrics(X=None,
                    Y=None, 
                    f=None, 
                    rho=None,
                    c=None):
    """Computes Evaluation metrics.
    
    Args:
        X: past-time samples as features, a float numpy array of shape (T, N, p).
        Y: target responses, a float numpy array of shape (T, N).
        f: forecast responses, a float numpy array of shape (T, N).
        rho: partial correlation parameters (symmetric), a float numpy array of shape (N, N).
        c: inverse of conditional variances, a float numpy array of shape (N, ).
    
    Returns:
        mse: mean squared error, float scalar.
        rsquared: correlation R^2, float scalar.
        frsquared: forecasting R^2, float scalar.
        
        Note we use the definition of R^2 in "Nets:  Network esti-mation  for  time  series" by M. Barigozzi and C. Brownlees.  
    """
    T, N = Y.shape
    eps = Y - f
    Theta = rho * np.sqrt(c / c[:, None])
    Ypred = f + eps@Theta.T
        
    fmse = mean_squared_error(Y.reshape(-1), f.reshape(-1), multioutput='uniform_average') 
    mse = mean_squared_error(Y.reshape(-1), Ypred.reshape(-1), multioutput='uniform_average') 
    f_rsquared = np.mean([1-mean_squared_error(Y[:,j], f[:,j])/mean_squared_error(Y[:,j], np.zeros(T)) for j in range(N)])
    rsquared = np.mean([1-mean_squared_error(Y[:,j], Ypred[:,j])/mean_squared_error(Y[:,j], np.zeros(T)) for j in range(N)])
    return fmse, mse, f_rsquared, rsquared
