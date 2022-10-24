import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
from copy import deepcopy, copy
from tqdm import notebook

def proximal_gradient_descent(X=None,
                              Y=None,
                              model=None,
                              eta=1e1,
                              max_iter=50000,
                              tol=1e-5
                              ):
    T, N = Y.shape
    
    
    J = np.inf
    
    # Get Neural Network prediction for forecasting component   
    layer_name = model.layers[1].name # fixed
    intermediate_layer_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    f = intermediate_layer_model.predict(x=[X, Y])
    eps = Y-f
    
    KG_params = model.layers[-1].get_weights()
    rho = KG_params[0]
    rho = 0.5*(rho+rho.T)
    rho_hat = model.layers[-1].partial_correlation_preestimator.numpy()
    M = model.layers[-1].M.numpy()
    masking_weight = np.where(np.isfinite(M), M, np.zeros_like(M))
    lam_2 = model.layers[-1].regularization_weight
    c = KG_params[1]
    c_shape = c.shape
    c = c.reshape(-1)
    rho_initial = deepcopy(rho)
        
    for it in range(max_iter):
        Theta = rho * np.sqrt(c / c[:, None])
        u = eps-eps@(Theta.T)

        J_prev = deepcopy(J)
        mse = np.linalg.norm(u,'fro')**2 / (N * T)
        J = mse + lam_2 * np.sum(np.absolute(
            np.multiply(
                np.divide(rho, rho_hat),
                masking_weight
            )
        ))
#         J1 = model.evaluate(x=[X, Y], y=Y, batch_size=Y.shape[0], verbose=0)[0]
#         print(J-J1)

        # Find gradients of J with respect to rho 
        grad_rho = (-np.sqrt(c / c[:, None]) * np.matmul(u.T, eps)*np.isfinite(M)) * 2 / (N * T)
        grad_rho += grad_rho.transpose()    
        np.fill_diagonal(grad_rho, 0)

        rho_bar = rho-eta*grad_rho
        # Update rho with soft-thresholding operator (proximal gradient update) for Lasso/Adaptive Lasso
        rho = np.maximum(
            (np.absolute(rho_bar) - eta * lam_2 * (np.absolute(M)/np.absolute(rho_hat))), 
            np.zeros_like(rho_bar)
        ) * np.sign(rho_bar)

#         KG_params[0] = rho
#         KG_params[1] = c.reshape(c_shape)
#         model.layers[-1].set_weights(KG_params)
            
        
        J_del = J - J_prev
        sparsity = (np.count_nonzero(rho)/(N*N))*100
        if it%100==0 or J>J_prev:
            print('\lambda-rho:{:.8f}, Iteration:{}, J:{:.5f}, |\Delta J /J|:{:.8f}, sparsity:{:.2f}'.format(lam_2, it, J, np.absolute(J_del/J), sparsity))
            if (J_del/np.absolute(J))>1e-5: # fix for model.evaluate with batch!
                J = np.inf
                rho = deepcopy(rho_initial)
                eta = 0.5*eta   
                print('WARNING!!!! Objective increased!!! Restarting from initial solution with learning rate {}. '.format(eta))
                continue
        if np.absolute(J_del/J)<tol or np.isnan(J):
            print('\lambda-rho:{:.8f}, Iteration:{}, J:{:.5f}, |\Delta J /J|:{:.8f}, sparsity:{:.2f}'.format(lam_2,it, J, np.absolute(J_del/J), sparsity))
            break

    KG_params[0] = rho
    KG_params[1] = c.reshape(c_shape)
    model.layers[-1].set_weights(KG_params)
           
    return model