import numpy as np
from scipy import sparse as sp
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf

def compute_metrics_with_pseudolikelihood(X=None,
                                          Y=None, 
                                          model=None):
    """Computes Evaluation metrics.
    
    Args:
        X: past-time samples as features, a float numpy array of shape (T, N, p).
        Y: target responses, a float numpy array of shape (T, N).
        model: keras model.
    
    Returns:
        mse: mean squared error, float scalar.
        rsquared: correlation R^2, float scalar.
        frsquared: forecasting R^2, float scalar.
        
        Note we use the definition of R^2 in "Nets:  Network estimation  for  time  series" by M. Barigozzi and C. Brownlees.  
    """
    
    Y_pred = model.predict([X, Y],
                           verbose=False)
    if np.sum(np.isnan(Y_pred))==0:
        forecast_model = tf.keras.Model(inputs=model.input, outputs=model.layers[1].output)
        f = forecast_model.predict([X, Y])
        mse = mean_squared_error(Y.reshape(-1), Y_pred.reshape(-1), multioutput='uniform_average') 
        # rsquared = r2_score(Y.reshape(-1), (P-Q+S).reshape(-1),multioutput='uniform_average')
        rsquared = np.mean([1-mean_squared_error(Y[:,j], Y_pred[:,j])/mean_squared_error(Y[:,j], np.zeros(Y.shape[0])) for j in range(Y.shape[1])])
        frsquared = np.mean([1-mean_squared_error(Y[:,j], f[:,j])/mean_squared_error(Y[:,j], np.zeros(Y.shape[0])) for j in range(Y.shape[1])])
    else:
        mse = np.inf
        rsquared = -np.inf
        frsquared = -np.inf
        
    return mse, rsquared, frsquared

def compute_metrics(X=None,
                    Y=None, 
                    model=None):
    """Computes Evaluation metrics.
    
    Args:
        X: past-time samples as features, a float numpy array of shape (T, N, p).
        Y: target responses, a float numpy array of shape (T, N).
        model: keras model.
    
    Returns:
        mse: mean squared error, float scalar.
        rsquared: correlation R^2, float scalar.
        
        Note we use the definition of R^2 in "Nets:  Network estimation  for  time  series" by M. Barigozzi and C. Brownlees.  
    """
    
    Y_pred = model.predict(X,
                           verbose=False)
    if np.sum(np.isnan(Y_pred))==0:
        Y_pred = np.squeeze(Y_pred)
        mse = mean_squared_error(Y.reshape(-1), Y_pred.reshape(-1), multioutput='uniform_average') 
        rsquared = np.mean([1-mean_squared_error(Y[:,j], Y_pred[:,j])/mean_squared_error(Y[:,j], np.zeros(Y.shape[0])) for j in range(Y.shape[1])])
    else:
        mse = np.inf
        rsquared = -np.inf        
    return mse, rsquared

def compute_all_metrics(X=None,
                        Y=None, 
                        model=None,
                        eps=1e-7):
    """Computes Evaluation metrics.
    
    Args:
        X: past-time samples as features, a float numpy array of shape (T, N, p).
        Y: target responses, a float numpy array of shape (T, N).
        model: keras model.
    
    Returns:
        mse: mean squared error, float scalar.
        mae: mean absolute error, float scalar.
        mape: mean absolute percentage error, float scalar.
        rsquared: correlation R^2, float scalar.        
    """
    
    Y_pred = model.predict(X)
    if np.sum(np.isnan(Y_pred))==0:
        Y_pred = np.squeeze(Y_pred)
        # mean_squared_error
        mse = mean_squared_error(Y, Y_pred, multioutput='uniform_average') 
        # mean_absolute_error
        mae = mean_absolute_error(Y, Y_pred, multioutput='uniform_average')
        # mean_absolute_percentage_error
        mask = np.abs(Y)>eps
        mask = mask/mask.sum(axis=0, keepdims=True)
        mape = mean_absolute_percentage_error(Y, Y_pred, sample_weight=mask, multioutput='uniform_average') 
        # R^2
        rsquared = r2_score(Y, Y_pred, multioutput='uniform_average') 
    else:
        mse = np.inf
        mae = np.inf
        mape = np.inf
        rsquared = -np.inf        
    return mse, mae, mape, rsquared
    
def masked_matmul(A,B,W):
    row,col,_= sp.find(W)
    # Get the sum-reduction using valid rows and corresponding cols from A, B
    out = np.einsum('ij,ji->i',A[row],B[:,col])
    # Store as sparse matrix
    out_sparse = sp.coo_matrix((out, (row, col)), shape=W.shape)
    return out_sparse