from __future__ import division, print_function
import os
import sys
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.
import argparse
from contextlib import redirect_stdout
from tqdm import notebook
from tqdm.keras import TqdmCallback
import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from IPython.display import Math
import seaborn as sb
import pandas as pd
import time
import pathlib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import clone_model
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute()).split('src')[0]))
from src.NNPC.Models import models
from src.NNPC import pre_estimators_NN
from src.NNPC import PartialCorrelation_optimizer
from src import data_utils
from src import graph_utils
from src import metrics

parser = argparse.ArgumentParser(description='LSTM with partial correlation using Knowledge Graph.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/nfs/sloanlab003/projects/FinancialForecasting_proj/data/')
parser.add_argument('--cohort', dest='cohort',  type=str, default='SP500')
parser.add_argument('--time_series', dest='time_series',  type=str, default='volatilities')

parser.add_argument('--num_training_years', dest='num_training_years',  type=int, default=2)
parser.add_argument('--sample_complexity', dest='sample_complexity',  type=str, default='small')
parser.add_argument('--seed', dest='seed',  type=int, default=8)

# Algorithm Arguments
parser.add_argument('--epochs', dest='epochs',  type=int, default=10000)
parser.add_argument('--ntrials', dest='ntrials',  type=int, default=500)
parser.add_argument('--patience', dest='patience',  type=int, default=20)
parser.add_argument('--post_convergence_tolerance', dest='post_convergence_tolerance',  type=float, default=1e-5)
parser.add_argument('--regularizer', dest='regularizer',  type=str, default='Lasso') # 'Lasso' or 'AdaptiveLasso'.
parser.add_argument('--knowledge_graph', dest='knowledge_graph',  type=str, default='cosearch') # 'cosearch' or 'returns_PC_KNN'
parser.add_argument('--mask', dest='mask', action='store_true')
parser.add_argument('--no-mask', dest='mask', action='store_false')
parser.set_defaults(mask=False)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=0)
parser.add_argument('--hp_tuning_file', dest='hp_tuning_file',  type=str, default='Hyperparameter_Tuning.txt')
parser.add_argument('--results_file', dest='results_file',  type=str, default='results.txt')
parser.add_argument('--settings_file', dest='settings_file',  type=str, default='Settings.txt')

# Debugging Arguments
parser.add_argument('--debugging_file', dest='debugging_file',  type=str, default='Warning.txt')

# parser.add_argument('--debug', dest='debug', action='store_true')

# Leave at default, updated automatically based on cohort selection
parser.add_argument('--knowledgegraph_dir', dest='knowledgegraph_dir',  type=str)
parser.add_argument('--knowledgegraph_ticker_file', dest='knowledgegraph_ticker_file',  type=str)
parser.add_argument('--residual_file', dest='residual_file',  type=str)
parser.add_argument('--log_dir', dest='log_dir',  type=str)

args = parser.parse_args()
args.knowledgegraph_dir = 'EDGAR/Stanford-'+args.cohort
if args.cohort == 'SP1500':
    args.knowledgegraph_ticker_file = 'cik_ticker_sp1500.csv'
elif args.cohort == 'SP500':
    args.knowledgegraph_ticker_file = 'cik_ticker_sp500.csv'
args.residual_file = os.path.join(args.time_series, args.cohort, "yahoo_companies_{}_residuals_2Y.csv".format(args.time_series))
args.log_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'logs-'+args.cohort)


if args.mask:
    args.KG_mask = ["soft"]
else:
    args.KG_mask = [None]

args.lstm_preestimator_params = {'n_steps': 15, 'optimizer': 'adam', 'lr': 0.001, 
                                 'batch_size': 64, 'seed': 10,
                                 'conv_layers': 1, 'conv_filtersize': 5, 'conv_filters': 300, 'conv_activation': 'relu', 
                                 'conv_dropout': 0.35,
                                 'lstm_layers': 1, 'lstm_units': 1024, 'lstm_recurrent_activation': 'sigmoid',
                                 'lstm_activation': 'tanh', 
                                 'dense_layers': 1, 'dense_units': 1024, 'dense_activation': 'relu',
                                 'kernel_regularizer': tf.keras.regularizers.l2(1e-5),  
                                 'output_activation': 'linear', 'use_bias': True} 
    
log_folder = """MODEL{}_COHORT{}_TS{}_REG{}_KG{}_MASK{}_Y{}_NTRIALS{}_V{}""".format(
    'LSTMPC', args.cohort, args.time_series, args.regularizer, args.knowledge_graph, args.mask, args.num_training_years, args.ntrials, args.version)

path = os.path.join(args.log_dir, log_folder)
os.makedirs(path, exist_ok=True)

with open(os.path.join(path, args.settings_file), 'w') as f:
    with redirect_stdout(f):
        for arg in vars(args):
            print(arg, '=', getattr(args, arg))    
            
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
            
df_train, df_val, df_test, cf_train, cf_val, cf_test, _ = data_utils.load_data(
    load_dir=args.load_directory, 
    folder=args.knowledgegraph_dir, 
    cosearch_ticker_file=args.knowledgegraph_ticker_file,
    residual_file = args.residual_file,
    N_years = args.num_training_years
)

df_train.head()

cf_train = 0.5*(cf_train+cf_train.transpose())
cf_val = 0.5*(cf_val+cf_val.transpose())
cf_test = 0.5*(cf_test+cf_test.transpose())
N = cf_train.shape[0]

def get_optimizer(opt,lr):
    if opt=='adam':
        optimizer = Adam(lr=lr)
    elif opt=='rmsprop':
        optimizer = RMSprop(learning_rate=lr, clipvalue=10)
    elif opt=='sgd':
        optimizer = SGD(lr=lr, momentum=0.1, nesterov=True, clipvalue=10)
    return optimizer


# Pre-estimators for NN, rho, c, M
# # convert into input/output
# # reshape input to be 3D [samples, timesteps, features]
X, Y = data_utils.prepare_sequences(df_train.values, args.lstm_preestimator_params['n_steps'])
X_val, Y_val = data_utils.prepare_sequences(np.append(df_train.values[-args.lstm_preestimator_params['n_steps']:],df_val.values, axis=0), args.lstm_preestimator_params['n_steps'])
X_test, Y_test = data_utils.prepare_sequences(np.append(df_val.values[-args.lstm_preestimator_params['n_steps']:],df_test.values, axis=0), args.lstm_preestimator_params['n_steps'])
print(X.shape, Y.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

X = np.swapaxes(X,1,2)
X_val = np.swapaxes(X_val,1,2) 
X_test = np.swapaxes(X_test,1,2)
print(X.shape, Y.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

callbacks = [EarlyStopping(monitor='val_mse', patience=args.patience, restore_best_weights=True)]
model_hat = models.create_lstm_model(
    input_shape=X.shape[1:],
    output_units=Y.shape[1], 
    conv_layers=args.lstm_preestimator_params['conv_layers'],
    conv_filtersize=args.lstm_preestimator_params['conv_filtersize'],
    conv_filters=args.lstm_preestimator_params['conv_filters'],
    conv_activation=args.lstm_preestimator_params['conv_activation'],
    conv_dropout=args.lstm_preestimator_params['conv_dropout'],
    conv_kernel_regularizer=args.lstm_preestimator_params['kernel_regularizer'],
    conv_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=args.lstm_preestimator_params['seed']),
    lstm_layers=args.lstm_preestimator_params['lstm_layers'],
    lstm_units=args.lstm_preestimator_params['lstm_units'],
    lstm_recurrent_activation=args.lstm_preestimator_params['lstm_recurrent_activation'],
    lstm_activation=args.lstm_preestimator_params['lstm_activation'],
    lstm_recurrent_regularizer=args.lstm_preestimator_params['kernel_regularizer'],
    lstm_kernel_regularizer=args.lstm_preestimator_params['kernel_regularizer'],
    lstm_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=args.lstm_preestimator_params['seed']),
    lstm_recurrent_initializer=tf.keras.initializers.Orthogonal(seed=args.lstm_preestimator_params['seed']),
    dense_layers=args.lstm_preestimator_params['dense_layers'],
    dense_units=args.lstm_preestimator_params['dense_units'],
    dense_kernel_regularizer=args.lstm_preestimator_params['kernel_regularizer'],
    dense_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=args.lstm_preestimator_params['seed']),
    dense_activation=args.lstm_preestimator_params['dense_activation'],
    use_bias=args.lstm_preestimator_params['use_bias'],
)

model_hat.save_weights(os.path.join(path, 'model_initial.h5'), overwrite=True)
model_hat.compile(loss='mean_squared_error', optimizer=get_optimizer(args.lstm_preestimator_params['optimizer'], args.lstm_preestimator_params['lr']), metrics=['mse'])
model_hat.fit(X, 
              Y, 
              epochs=args.epochs, 
              batch_size=args.lstm_preestimator_params['batch_size'], 
              shuffle=True,
              callbacks=callbacks,
              validation_data=(X_val, Y_val), 
              verbose=1, 
              )

space = {'epochs' :  args.epochs,
         'optimizer': hp.choice('optimizer',['adam']),
         'batch_size' : hp.choice('batch_size', [32, 64, 128, 256]),
         'lr': hp.choice('lr', np.logspace(start=-4, stop=-2, num=20, base=10)),
         'n_steps' : hp.choice('n_steps', [5, 10, 15, 20]),
         'use_bias': hp.choice('use_bias', [True, False]),
         'conv_layers': hp.choice('conv_layers', [0, 1]),
         'conv_filtersize': hp.choice('conv_filtersize', [3, 5, 7, 9]),
         'conv_filters': hp.choice('conv_filters', [256, 512, 1024]),
         'conv_dropout': hp.choice('conv_dropout', np.linspace(start=0.05, stop=0.5, num=10)),
         'lstm_layers': hp.choice('lstm_layers', [1]),
         'lstm_units': hp.choice('lstm_units', [512, 1024, 1536]),
         'lstm_recurrent_activation': 'sigmoid',
         'lstm_activation': 'tanh',
         'dense_layers': hp.choice('dense_layers', [1]),
         'dense_units': hp.choice('dense_units', [512, 1024, 1536]),
         'dense_activation': 'relu',
         'kernel_regularizer': hp.choice('kernel_regularizer', np.logspace(start=-6, stop=-3, num=10, base=10)),
         'output_activation': 'linear',
         'seed_initializer': 10,
         'save_file': os.path.join(path, args.hp_tuning_file),
         'knowledge_graph_mask_rho': hp.choice('knowledge_graph_mask_rho', args.KG_mask),
         'knowledge_graph_mask_sparsity_rho': hp.choice('knowledge_graph_mask_sparsity_rho', [100, 150, 200, 250, 300, 350, 400]),
         'regularizer_rho': hp.choice('regularizer_rho', ['Lasso']),
         'lambda_rho': hp.choice('lambda_rho', np.logspace(start=-3, stop=-7, num=20, base=10.0)),
         'post_learning_rate_rho': hp.choice('post_learning_rate_rho', [1e-0]),
        }
 
def get_optimizer(opt, lr):
    if opt=='adam':
        optimizer = Adam(lr=lr)
    elif opt=='rmsprop':
        optimizer = RMSprop(learning_rate=lr, clipvalue=10)
    elif opt=='sgd':
        optimizer = SGD(lr=lr, momentum=0.1, nesterov=True, clipvalue=10)
    return optimizer


def f_model(params):
    # # convert into input/output
    # # reshape input to be 3D [samples, timesteps, features]
    X, Y = data_utils.prepare_sequences(df_train.values, params['n_steps'])
    X_val, Y_val = data_utils.prepare_sequences(np.append(df_train.values[-params['n_steps']:], df_val.values, axis=0), params['n_steps'])

    X = np.swapaxes(X,1,2)
    X_val = np.swapaxes(X_val,1,2) 
        
    # Get pre-estimators
    c, _, rho_hat = pre_estimators_NN.get_pre_estimators(X=X, Y=Y, model=model_hat, pen=params['regularizer_rho'], n=args.sample_complexity)
    M_train = pre_estimators_NN.get_M(cf=cf_train,
                                      masking=params['knowledge_graph_mask_rho'],
                                      sparse=params['knowledge_graph_mask_sparsity_rho'],
                                      alpha=0)
    
    # Create Model with Pseudolikelihood Layer
    model = models.create_model_with_pseudolikelihood(
        models.create_lstm_model,
        {'input_shape': X.shape[1:],
         'output_units': Y.shape[1],
         'conv_layers': params['conv_layers'],
         'conv_filtersize': params['conv_filtersize'],
         'conv_filters': params['conv_filters'],
         'conv_dropout': params['conv_dropout'],
         'conv_kernel_regularizer': tf.keras.regularizers.l2(params['kernel_regularizer']),
         'conv_kernel_initializer': tf.keras.initializers.GlorotUniform(seed=params['seed_initializer']),
         'lstm_layers': params['lstm_layers'],
         'lstm_units': params['lstm_units'],
         'lstm_recurrent_activation': params['lstm_recurrent_activation'],
         'lstm_activation': params['lstm_activation'],
         'lstm_recurrent_regularizer': tf.keras.regularizers.l2(params['kernel_regularizer']),
         'lstm_kernel_regularizer': tf.keras.regularizers.l2(params['kernel_regularizer']),
         'lstm_kernel_initializer': tf.keras.initializers.GlorotUniform(seed=params['seed_initializer']),
         'lstm_recurrent_initializer': tf.keras.initializers.Orthogonal(seed=params['seed_initializer']),
         'dense_layers': params['dense_layers'],
         'dense_units': params['dense_units'],
         'dense_kernel_regularizer': tf.keras.regularizers.l2(params['kernel_regularizer']),
         'dense_kernel_initializer': tf.keras.initializers.GlorotUniform(seed=params['seed_initializer']),
         'dense_activation': params['dense_activation'],
         'use_bias': params['use_bias'],
         },
        knowledge_graph_weighted_mask=M_train,
        partial_correlation_preestimator=rho_hat,
        c=c.reshape(-1),
        partial_correlation_regularization_weight=params['lambda_rho'],
    )        

    model.compile(loss='mean_squared_error', 
                  optimizer=get_optimizer(params['optimizer'], params['lr']), 
                  metrics=['mse'])

    # Train model
    callbacks = [EarlyStopping(monitor='val_mse', patience=args.patience, restore_best_weights=True), TerminateOnNaN()]    
    params_dict = deepcopy(params)
    params_dict.pop('save_file', None)
    with open(params['save_file'], 'a') as f:
        with redirect_stdout(f):
            print(params_dict)
    print ('Params testing: ', params_dict)
    model.fit([X, Y],
              Y, 
              epochs=params['epochs'], 
              batch_size=params['batch_size'], 
              shuffle=True,
              callbacks=callbacks,
              validation_data=([X_val, Y_val], Y_val), 
              verbose=0, 
              ) 
    
    loss = model.evaluate([X, Y], Y, verbose=0)[0]
    print(loss)
    if np.isfinite(loss) or ~np.isnan(loss):
        model = PartialCorrelation_optimizer.proximal_gradient_descent(
            X=X, Y=Y,
            model=model,
            eta=params['post_learning_rate_rho'],
            tol=args.post_convergence_tolerance,
        )
        KG_params = model.layers[-1].get_weights()
        rho = KG_params[0]
        c =  KG_params[1]

        # Evaluate model
        MSE, Rsquared, _ = metrics.compute_metrics_with_pseudolikelihood(X=X, Y=Y, model=model)
        MSE_val, Rsquared_val, _ = metrics.compute_metrics_with_pseudolikelihood(X=X_val, Y=Y_val, model=model)

        N = Y.shape[-1]
        sparsity = (np.count_nonzero(rho)/(N*N))*100
    else:
        MSE = np.inf
        MSE_val = np.inf
        Rsquared = -np.inf
        Rsquared_val = -np.inf
        sparsity = 100
        c =  1/np.var(Y, axis=0)
    with open(params['save_file'], 'a') as f:
        with redirect_stdout(f):
            print('MSE:{:.5f}, R^2:{:.5f}, MSE-val:{:.5f}, R^2-val:{:.5f}, sparsity:{:.2f}\n'.format(MSE, Rsquared, MSE_val, Rsquared_val, sparsity))        
    print('MSE:{:.5f}, R^2:{:.5f}, MSE-val:{:.5f}, R^2-val:{:.5f}, sparsity:{:.2f}'.format(MSE, Rsquared, MSE_val, Rsquared_val, sparsity))        
    sys.stdout.flush()
    return {'loss': MSE_val, 'status': STATUS_OK, 'c': c}


with open(os.path.join(path, args.hp_tuning_file), 'w') as f:
    with redirect_stdout(f):
        print('Starting Hyperparameter Tuning')
start = time.time()

trials = Trials()
best_run = fmin(f_model, 
                space, 
                algo=tpe.rand.suggest, 
                max_evals=args.ntrials, 
                trials=trials, 
                return_argmin=False)

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
with open(os.path.join(path, args.results_file), "w") as f:
    with redirect_stdout(f):
        print("Training completed in {:0>2}:{:0>2}:{:05.2f} for {} hyperparameter settings.\n".format(int(hours),int(minutes),seconds,args.ntrials))
print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds, args.ntrials)) 

c_opt = trials.results[np.argmin([r['loss'] for r in trials.results])]['c']
# best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
# best_model.summary()
# best_model.save_weights(os.path.join(path, 'weights_opt_val.h5'), overwrite=True)


X, Y = data_utils.prepare_sequences(df_train.values, args.lstm_preestimator_params['n_steps'])
X_val, Y_val = data_utils.prepare_sequences(np.append(df_train.values[-args.lstm_preestimator_params['n_steps']:],df_val.values, axis=0), args.lstm_preestimator_params['n_steps'])

X = np.swapaxes(X,1,2)
X_val = np.swapaxes(X_val,1,2) 
# MSE, Rsquared, _ = metrics.compute_metrics_with_pseudolikelihood(X=X, Y=Y, model=best_model)
# MSE_val, Rsquared_val, _ = metrics.compute_metrics_with_pseudolikelihood(X=X_val, Y=Y_val, model=best_model)
# N = Y.shape[-1]
# rho_opt = best_model.layers[-1].get_weights()[0]
# c_opt = best_model.layers[-1].get_weights()[1]
# sparsity = (np.count_nonzero(rho_opt)/(N*N))*100 
best_run_dict = deepcopy(best_run)
best_run_dict.pop('save_file', None)
with open(os.path.join(path, args.results_file), 'w') as f:
    with redirect_stdout(f):
        print(best_run_dict)
#         print('MSE:{:.5f}, R^2:{:.5f}, MSE-val:{:.5f}, R^2-val:{:.5f}, sparsity:{:.2f}'.format(MSE, Rsquared, MSE_val, Rsquared_val, sparsity))        


# plt.figure(figsize=(13,10))
# sb.heatmap(np.absolute(rho_opt[np.ix_(np.arange(100),np.arange(100))]), xticklabels=False, yticklabels=False, cmap="Blues")
# plt.title('Partial correlation')
# plt.show()


# ## Run with optimal hyperparameters on training+validation data and evaluate performance on test data
df_train_val = pd.concat([df_train, df_val], axis=0)
X_train_val, Y_train_val = data_utils.prepare_sequences(df_train_val.values, best_run['n_steps'])
X_train_val = np.swapaxes(X_train_val,1,2)
X_test, Y_test = data_utils.prepare_sequences(np.append(df_val.values[-best_run['n_steps']:],df_test.values, axis=0), best_run['n_steps'])
X_test = np.swapaxes(X_test,1,2)
print(X_train_val.shape, Y_train_val.shape, X_test.shape, Y_test.shape)


# MSE_train_val, Rsquared_train_val, _ = metrics.compute_metrics_with_pseudolikelihood(X=X_train_val, Y=Y_train_val, model=best_model)
# MSE_test, Rsquared_test, _ = metrics.compute_metrics_with_pseudolikelihood(X=X_test, Y=Y_test, model=best_model)
# print('MSE:{:.5f}, R^2:{:.5f}, MSE-test:{:.5f}, R^2-test:{:.5f}'.format(MSE_train_val, Rsquared_train_val, MSE_test, Rsquared_test))        


model_hat = models.create_lstm_model(
    input_shape=X.shape[1:],
    output_units=Y.shape[1], 
    conv_layers=args.lstm_preestimator_params['conv_layers'],
    conv_filtersize=args.lstm_preestimator_params['conv_filtersize'],
    conv_filters=args.lstm_preestimator_params['conv_filters'],
    conv_activation=args.lstm_preestimator_params['conv_activation'],
    conv_dropout=args.lstm_preestimator_params['conv_dropout'],
    conv_kernel_regularizer=args.lstm_preestimator_params['kernel_regularizer'],
    conv_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=args.lstm_preestimator_params['seed']),
    lstm_layers=args.lstm_preestimator_params['lstm_layers'],
    lstm_units=args.lstm_preestimator_params['lstm_units'],
    lstm_recurrent_activation=args.lstm_preestimator_params['lstm_recurrent_activation'],
    lstm_activation=args.lstm_preestimator_params['lstm_activation'],
    lstm_recurrent_regularizer=args.lstm_preestimator_params['kernel_regularizer'],
    lstm_kernel_regularizer=args.lstm_preestimator_params['kernel_regularizer'],
    lstm_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=args.lstm_preestimator_params['seed']),
    lstm_recurrent_initializer=tf.keras.initializers.Orthogonal(seed=args.lstm_preestimator_params['seed']),
    dense_layers=args.lstm_preestimator_params['dense_layers'],
    dense_units=args.lstm_preestimator_params['dense_units'],
    dense_kernel_regularizer=args.lstm_preestimator_params['kernel_regularizer'],
    dense_kernel_initializer=tf.keras.initializers.GlorotUniform(seed=args.lstm_preestimator_params['seed']),
    dense_activation=args.lstm_preestimator_params['dense_activation'],
    use_bias=args.lstm_preestimator_params['use_bias'],
)
model_hat.save_weights(os.path.join(path, 'model_initial.h5'), overwrite=True)
model_hat.compile(loss='mean_squared_error', optimizer=get_optimizer(args.lstm_preestimator_params['optimizer'], args.lstm_preestimator_params['lr']), metrics=['mse'])
model_hat.fit(X, 
              Y, 
              epochs=args.epochs, 
              batch_size=args.lstm_preestimator_params['batch_size'], 
              shuffle=True,
              callbacks=callbacks,
              validation_data=(X_val, Y_val), 
              verbose=0, 
              )

# model_hat.compile(loss='mean_squared_error', optimizer=optimizer, metrics = ['mse'])
# model_hat.load_weights(os.path.join(path, 'model_initial.h5'), overwrite=True)
# model_hat.fit(X_train_val, 
#               Y_train_val, 
#               epochs=args.epochs, 
#               batch_size=batch_size, 
#               shuffle=True,
#               callbacks=callbacks,
# #               validation_data=(X_test, Y_test), 
#               verbose=1, 
#               )

_, _, rho_hat = pre_estimators_NN.get_pre_estimators(X=X_train_val, Y=Y_train_val, model=model_hat, pen=best_run['regularizer_rho'], n=args.sample_complexity)
M_train_val = pre_estimators_NN.get_M(cf=cf_val, masking=best_run['knowledge_graph_mask_rho'], sparse=best_run['knowledge_graph_mask_sparsity_rho'], alpha=0)


model_opt = models.create_model_with_pseudolikelihood(
    models.create_lstm_model,
    {'input_shape': X_train_val.shape[1:],
     'output_units': Y.shape[1],
     'conv_layers': best_run['conv_layers'],
     'conv_filtersize': best_run['conv_filtersize'],
     'conv_filters': best_run['conv_filters'],
     'conv_dropout': best_run['conv_dropout'],
     'conv_kernel_regularizer': tf.keras.regularizers.l2(best_run['kernel_regularizer']),
     'conv_kernel_initializer': tf.keras.initializers.GlorotUniform(seed=best_run['seed_initializer']),
     'lstm_layers': best_run['lstm_layers'],
     'lstm_units': best_run['lstm_units'],
     'lstm_recurrent_activation': best_run['lstm_recurrent_activation'],
     'lstm_activation': best_run['lstm_activation'],
     'lstm_recurrent_regularizer': tf.keras.regularizers.l2(best_run['kernel_regularizer']),
     'lstm_kernel_regularizer': tf.keras.regularizers.l2(best_run['kernel_regularizer']),
     'lstm_kernel_initializer': tf.keras.initializers.GlorotUniform(seed=best_run['seed_initializer']),
     'lstm_recurrent_initializer': tf.keras.initializers.Orthogonal(seed=best_run['seed_initializer']),
     'dense_layers': best_run['dense_layers'],
     'dense_units': best_run['dense_units'],
     'dense_kernel_regularizer': tf.keras.regularizers.l2(best_run['kernel_regularizer']),
     'dense_kernel_initializer': tf.keras.initializers.GlorotUniform(seed=best_run['seed_initializer']),
     'dense_activation': best_run['dense_activation'],
     'use_bias': best_run['use_bias'],
    },
    knowledge_graph_weighted_mask=M_train_val,
    partial_correlation_preestimator=rho_hat,
    c=c_opt.reshape(-1),
    partial_correlation_regularization_weight=best_run['lambda_rho'],
)        

model_opt.compile(loss='mean_squared_error', optimizer=get_optimizer(best_run['optimizer'], best_run['lr']), metrics=['mse'])
# model_opt.load_weights(os.path.join(path, 'weights_opt.h5'))


callbacks = [EarlyStopping(monitor='val_mse', patience=args.patience, restore_best_weights=True), TerminateOnNaN()]
model_opt.fit([X_train_val, Y_train_val], 
              Y_train_val, 
              epochs=args.epochs, 
              batch_size=best_run['batch_size'], 
              shuffle=True,
              verbose=0, 
              callbacks=callbacks,
              validation_data=([X_test, Y_test], Y_test)
              )  

model_opt = PartialCorrelation_optimizer.proximal_gradient_descent(
    X=X_train_val,
    Y=Y_train_val,
    model=model_opt,
    eta=best_run['post_learning_rate_rho'],
    tol=args.post_convergence_tolerance,
)
KG_params = model_opt.layers[-1].get_weights()
rho_opt = KG_params[0]
c_opt =  KG_params[1]

MSE_train_val, Rsquared_train_val, _ = metrics.compute_metrics_with_pseudolikelihood(X=X_train_val, Y=Y_train_val, model=model_opt)
MSE_test, Rsquared_test, _ = metrics.compute_metrics_with_pseudolikelihood(X=X_test, Y=Y_test, model=model_opt)
sparsity = (np.count_nonzero(rho_opt)/(N*N))*100
with open(os.path.join(path, args.results_file), 'w') as f:
    with redirect_stdout(f):
        print(best_run_dict)
        print('MSE:{:.5f}, R^2:{:.5f}, MSE-test:{:.5f}, R^2-test:{:.5f}, sparsity:{:.2f}'.format(MSE_train_val, Rsquared_train_val, MSE_test, Rsquared_test, sparsity))        
print('MSE:{:.5f}, R^2:{:.5f}, MSE-test:{:.5f}, R^2-test:{:.5f}, sparsity:{:.2f}'.format(MSE_train_val, Rsquared_train_val, MSE_test, Rsquared_test, sparsity))        
model_opt.save_weights(os.path.join(path, 'weights_opt.h5'), overwrite=True)
# model_opt.save(os.path.join(path, "model"))


# ### Visualize partial correlation
plt.figure(figsize=(13,10))
sb.heatmap(np.absolute(rho_opt[np.ix_(np.arange(100),np.arange(100))]), xticklabels=False, yticklabels=False, cmap="Blues")
plt.title('Partial correlation')
plt.show()