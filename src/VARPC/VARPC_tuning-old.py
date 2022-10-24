from __future__ import division, print_function
import os
import sys
import argparse
from contextlib import redirect_stdout

print(sys.version, sys.platform, sys.executable)
import pandas as pd
from tqdm import notebook
import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from IPython.display import Math, display
import seaborn as sb
import time
import pathlib
import dill

sys.path.insert(0, os.path.abspath(os.path.join(pathlib.Path(__file__).parent.absolute(), os.pardir)))

import models
import utilities

parser = argparse.ArgumentParser(description='VAR with partial correlation using Knowledge Graph.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/nfs/sloanlab003/projects/FinancialForecasting_proj/data/')
parser.add_argument('--cohort', dest='cohort',  type=str, default='SP500')
parser.add_argument('--time_series', dest='time_series',  type=str, default='volatilities')

parser.add_argument('--num_training_years', dest='num_training_years',  type=int, default=2)
parser.add_argument('--sample_size', dest='sample_size',  type=str, default='small')
parser.add_argument('--seed', dest='seed',  type=int, default=8)
parser.add_argument('--n_steps', dest='n_steps',  type=int, default=1) # corresponds to p.

# Algorithm Arguments
parser.add_argument('--max_iter', dest='max_iter',  type=int, default=100000)
parser.add_argument('--regularizer', dest='regularizer',  type=str, default='Lasso') # 'Lasso' or 'AdaptiveLasso', or 'L0L2'.
parser.add_argument('--knowledge_graph', dest='knowledge_graph',  type=str, default='cosearch') # 'cosearch' or 'returns_PC_KNN'
parser.add_argument('--KG_mask', dest='KG_mask',  type=str, default=None) # None or 'hard' or 'soft'.
parser.add_argument('--mask_sparsity', dest='mask_sparsity',  type=int, default=None)

# Tuning Arguments
parser.add_argument('--tuning_metric', dest='tuning_metric',  type=str, default='mse') # fmse, mse

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
if args.time_series == "volume":
    args.pre_estimator_learning_rate = 1.0
    args.pre_estimator_lam = 1e-4
    args.learning_rate = 2.5e-1
    args.convergence_tolerance = 1e-5
elif args.time_series == "volatilities":
    args.pre_estimator_learning_rate = 10.0
    args.pre_estimator_lam = 1e-3
    args.learning_rate = 2.5
    args.convergence_tolerance = 1e-5

if args.KG_mask=="None":
    args.KG_mask = None

log_folder = """MODEL{}_COHORT{}_TS{}_REG{}_KG{}_MASK{}_LEVEL{}_Y{}_NSTEPS{}_TUNING{}_V{}""".format(
    'VARPC', args.cohort, args.time_series, args.regularizer, args.knowledge_graph, args.KG_mask, args.mask_sparsity, args.num_training_years, args.n_steps, args.tuning_metric, args.version)

path = os.path.join(args.log_dir, log_folder)
os.makedirs(path, exist_ok=True)

with open(os.path.join(path, args.settings_file), 'w') as f:
    with redirect_stdout(f):
        for arg in vars(args):
            print(arg, '=', getattr(args, arg))

np.random.seed(args.seed)

df_train, df_val, df_test, cf_train, cf_val, cf_test, _ = utilities.load_data(
    load_dir=args.load_directory,
    folder=args.knowledgegraph_dir,
    cosearch_ticker_file=args.knowledgegraph_ticker_file,
    residual_file = args.residual_file,
    N_years = args.num_training_years
)

## Process data to define X, Y and symmetrize cosearch
# # reshape input to be 3D [samples, timesteps, features]
X, Y = utilities.prepare_sequences(df_train.values, args.n_steps)
X_val, Y_val = utilities.prepare_sequences(np.append(df_train.values[-args.n_steps:],df_val.values, axis=0), args.n_steps)
X_test, Y_test = utilities.prepare_sequences(np.append(df_val.values[-args.n_steps:],df_test.values, axis=0), args.n_steps)
print('(N, T, p):', X.shape)
print('Training Data Shapes:', X.shape, ',', Y.shape)
print('Validation Data Shapes:', X_val.shape, ',', Y_val.shape)
print('Test Data Shapes:', X_test.shape, ',', Y_test.shape)

if args.knowledge_graph == "cosearch":
    pass
elif args.knowledge_graph == "returns_PC_KNN":
    args.graph_folder = os.path.join("graphs/timeseries_related", args.cohort, "Clustering")
    cf_train = sp.load_npz(
        os.path.join(args.load_directory,
                     args.graph_folder,
                     "train_{}.npz".format(args.knowledge_graph)
        )
    ).toarray()
    cf_val = sp.load_npz(
        os.path.join(args.load_directory,
                     args.graph_folder,
                     "val_{}.npz".format(args.knowledge_graph)
        )
    ).toarray()
else:
    raise ValueError("Knowledge Graph {} is not supported".format(args.knowledge_graph))

cf_train = np.abs(cf_train)
cf_val = np.abs(cf_val)
cf_train = 0.5*(cf_train+cf_train.transpose())
cf_val = 0.5*(cf_val+cf_val.transpose())

varpc = models.VARPC(cf_train, args.regularizer, args.KG_mask, args.mask_sparsity, path=path)
varpc.compute_preestimators(X, Y, A_eta=args.pre_estimator_learning_rate, A_lam=args.pre_estimator_lam, sample_size=args.sample_size)

## Hyperparameter tuning
if args.regularizer not in ['Lasso', 'AdaptiveLasso', 'L0L2']:
    raise ValueError('Ragularizer {} is not supported.'.format(args.regularizer))
if args.KG_mask not in [None, 'hard', 'soft']:
    raise ValueError('Ragularizer {} is not supported.'.format(args.KG_mask))
if args.time_series == 'volume':
    lams_A = np.logspace(start=-6, stop=-9, num=20, base=10.0)
    if args.regularizer == 'Lasso':
        if args.KG_mask is None:
            lams_rho = np.logspace(start=-4, stop=-7, num=25, base=10.0).reshape(-1, 1)
        elif args.KG_mask == 'hard':
            lams_rho = np.logspace(start=-4, stop=-7, num=25, base=10.0).reshape(-1, 1)
        elif args.KG_mask == 'soft':
            lams_rho = np.logspace(start=-5, stop=-8, num=25, base=10.0).reshape(-1, 1)
    elif args.regularizer == 'AdaptiveLasso':
        if args.KG_mask is None:
            lams_rho = np.logspace(start=-4, stop=-7, num=25, base=10.0).reshape(-1, 1)
        elif args.KG_mask == 'hard':
            lams_rho = np.logspace(start=-5, stop=-8, num=25, base=10.0).reshape(-1, 1)
        elif args.KG_mask == 'soft':
            lams_rho = np.logspace(start=-7, stop=-10, num=25, base=10.0).reshape(-1, 1)
    elif args.regularizer == "L0L2":
        if args.KG_mask is None:
            lams_rho_L0 = np.logspace(start=-6, stop=-8, num=10, base=10.0)
        elif args.KG_mask == "hard":
            lams_rho_L0 = np.logspace(start=-6, stop=-8, num=10, base=10.0)
        elif args.KG_mask == 'soft':
            lams_rho_L0 = np.logspace(start=-7, stop=-9, num=10, base=10.0)
        lams_rho_L2 = np.logspace(start=-3, stop=-5, num=10, base=10.0)
        l0, l2 = np.meshgrid(lams_rho_L0, lams_rho_L2, indexing='xy') # switch indexing to 'xy' for warm-starts across L0
        lams_rho = np.vstack((np.ravel(l0), np.ravel(l2))).T
elif args.time_series == 'volatilities':
    lams_A = np.logspace(start=-4, stop=-7, num=20, base=10.0)
    if args.regularizer == 'Lasso':
        if args.KG_mask is None:
            lams_rho = np.logspace(start=-3, stop=-6, num=25, base=10.0).reshape(-1, 1)
        elif args.KG_mask == 'hard':
            lams_rho = np.logspace(start=-4, stop=-7, num=25, base=10.0).reshape(-1, 1)
        elif args.KG_mask == 'soft':
            lams_rho = np.logspace(start=-4, stop=-7, num=25, base=10.0).reshape(-1, 1)
    elif args.regularizer == 'AdaptiveLasso':
        if args.KG_mask is None:
            lams_rho = np.logspace(start=-4, stop=-7, num=25, base=10.0).reshape(-1, 1)
        elif args.KG_mask == 'hard':
            lams_rho = np.logspace(start=-5, stop=-8, num=25, base=10.0).reshape(-1, 1)
        elif args.KG_mask == 'soft':
            lams_rho = np.logspace(start=-6, stop=-9, num=25, base=10.0).reshape(-1, 1)
    elif args.regularizer == "L0L2":
        if args.KG_mask is None:
            lams_rho_L0 = np.logspace(start=-6, stop=-9, num=10, base=10.0)
        elif args.KG_mask == "hard":
            lams_rho_L0 = np.logspace(start=-6, stop=-9, num=10, base=10.0)
        elif args.KG_mask == 'soft':
            lams_rho_L0 = np.logspace(start=-7, stop=-10, num=10, base=10.0)
        lams_rho_L2 = np.logspace(start=-3, stop=-5, num=10, base=10.0)
        l0, l2 = np.meshgrid(lams_rho_L0, lams_rho_L2, indexing='xy') # switch indexing to 'xy' for warm-starts across L0
        lams_rho = np.vstack((np.ravel(l0), np.ravel(l2))).T

T, N, p = X.shape
eta = args.learning_rate
val_opt = np.inf
varpc.A = np.zeros((N, N, p),dtype=float)
varpc.rho = np.zeros((N, N), dtype=float)
with open(os.path.join(path, args.hp_tuning_file), 'w') as f:
    with redirect_stdout(f):
        print('{0}  {1}  {2}  {3}  {4}  {5}  {6}  {7}  {8}'.format('lambda_A', 'lambda_rho', 'fMSE', 'MSE', 'fMSE-val', 'MSE-val', 'fR^2-val', 'R^2-val', 'Sparsity'))
# print('\lambda_1, \lambda_2, MSE, R^2, MSE-val, R^2-val, Sparsity')
warm_starts=False
start = time.time()
for lambda_A in notebook.tqdm(lams_A, 'lambda_A'):
    for lambda_rho in notebook.tqdm(lams_rho, 'lambda_rho'):

        varpc.fit(X, Y, lambda_A, lambda_rho, eta, convergence_tolerance=args.convergence_tolerance, max_iter=args.max_iter, warm_starts=warm_starts)
        warm_starts = True
        fMSE, MSE, _, _, sparsity = varpc.evaluate(X, Y)
        fMSE_val, MSE_val, fRsquared_val, Rsquared_val, _ = varpc.evaluate(X_val, Y_val)
        if args.tuning_metric == 'fmse':
            val = deepcopy(fMSE_val)
        elif args.tuning_metric == 'mse':
            val = deepcopy(MSE_val)
        else:
            raise ValueError('Tuning criteria {} is not supported'.format(args.tuning_metric))
        with open(os.path.join(path, args.hp_tuning_file), 'a') as f:
            with redirect_stdout(f):
                print('{0:.12f}  {1}  {2:.5f}  {3:.5f}  {4:.5f}  {5:.5f}  {6:.5f}  {7:.5f}  {8:.2f}'.format(lambda_A, *(tuple(lambda_rho),), fMSE, MSE, fMSE_val, MSE_val, fRsquared_val, Rsquared_val, sparsity))      
        print('{0:.12f}  {1}  {2:.5f}  {3:.5f}  {4:.5f}  {5:.5f}  {6:.5f}  {7:.5f}  {8:.2f}'.format(lambda_A, *(tuple(lambda_rho),), fMSE, MSE, fMSE_val, MSE_val, fRsquared_val, Rsquared_val, sparsity))      

        if val<val_opt:
            val_opt = deepcopy(val)
            varpc_opt = deepcopy(varpc)
            fMSE_val_opt = deepcopy(fMSE_val)
            MSE_val_opt = deepcopy(MSE_val)
            fRsquared_val_opt = deepcopy(fRsquared_val)
            Rsquared_val_opt = deepcopy(Rsquared_val)
            sparsity_opt = deepcopy(sparsity)
            lambda_A_opt = deepcopy(lambda_A)
            lambda_rho_opt = deepcopy(lambda_rho)
end = time.time()

with open(os.path.join(path, args.results_file), 'w') as f:
    with redirect_stdout(f):
        print('Optimal: lambda_A:{:.12f}\n lambda_rho:{}\n fMSE-val:{:.5f}\n MSE-val:{:.5f}\n fR2-val:{:.5f}\n R2-val:{:.5f}, sparsity:{:.2f}\n\n'.format(lambda_A_opt, *(tuple(lambda_rho_opt),), fMSE_val_opt, MSE_val_opt, fRsquared_val_opt, Rsquared_val_opt, sparsity_opt)) 
display(Math(r'~Optimal: \lambda_A:{:.12f},~\lambda_\rho:{},~fMSE-val:{:.5f},~MSE-val:{:.5f},~fR^2-val:{:.5f},~R^2-val:{:.5f},~sparsity:{:.2f}'.format(lambda_A_opt, *(tuple(lambda_rho_opt),), fMSE_val_opt, MSE_val_opt, fRsquared_val_opt, Rsquared_val_opt, sparsity_opt)))

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
with open(os.path.join(path, args.results_file), "a") as f:
    with redirect_stdout(f):
        print("Training completed in {:0>2}:{:0>2}:{:05.2f} for {} hyperparameter settings.\n".format(int(hours),int(minutes),seconds,int(lams_A.shape[0]*lams_rho.shape[0])))
print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds, int(lams_A.shape[0]*lams_rho.shape[0])))


## Run with optimal hyperparameters on training+validation data and evaluate performance on test data

df_train_val = pd.concat([df_train, df_val], axis=0)
X_train_val, Y_train_val = utilities.prepare_sequences(df_train_val.values, args.n_steps)

varpc_final = models.VARPC(cf_val, args.regularizer, args.KG_mask, args.mask_sparsity, path=path)
varpc_final.compute_preestimators(X_train_val, Y_train_val, A_eta=args.pre_estimator_learning_rate, A_lam=args.pre_estimator_lam,  sample_size=args.sample_size)
# varpc_final.A = deepcopy(varpc_opt.A)
# varpc_final.rho = deepcopy(varpc_opt.rho)
# varpc_final.c = deepcopy(varpc_opt.c)
start = time.time()
varpc_final.fit(X_train_val, Y_train_val, lambda_A_opt, lambda_rho_opt, eta, max_iter=args.max_iter, convergence_tolerance=args.convergence_tolerance, warm_starts=False)
end = time.time()

fMSE_train_val, MSE_train_val, fRsquared_train_val, Rsquared_train_val, sparsity_final = varpc_final.evaluate(X_train_val, Y_train_val)
fMSE_test, MSE_test, fRsquared_test, Rsquared_test, _ = varpc_final.evaluate(X_test, Y_test)

with open(os.path.join(path, args.results_file), 'a') as f:
    with redirect_stdout(f):
        print('fMSE-test:{:.5f}\n MSE-test:{:.5f}\n fR2-test:{:.5f}\n R2-test:{:.5f}\n Sparsity:{:.2f}\n'.format(fMSE_test, MSE_test, fRsquared_test, Rsquared_test, sparsity_final))
display(Math(r'~Optimal: \lambda_A:{:.8f},~\lambda_\rho:{},~fMSE:{:.5f},~MSE:{:.5f},~fR^2:{:.5f},~R^2:{:.5f},~fMSE-test:{:.5f},~MSE-test:{:.5f},~fR^2-test:{:.5f},~R^2-test:{:.5f},~sparsity:{:.2f}'.format(lambda_A_opt, *(tuple(lambda_rho_opt),), fMSE_train_val, MSE_train_val, fRsquared_train_val, Rsquared_train_val, fMSE_test, MSE_test, fRsquared_test, Rsquared_test, sparsity_final)))

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
with open(os.path.join(path, args.results_file), "a") as f:
    with redirect_stdout(f):
        print("Training completed in {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))
print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


### Visualize partial correlation

rho_final = varpc_final.rho
c_final = varpc_final.c
A_final = varpc_final.A
# plt.figure(figsize=(13,10))
# sb.heatmap(np.absolute(rho_final[np.ix_(np.arange(100),np.arange(100))]), xticklabels=False, yticklabels=False, cmap="Blues")
# plt.title('Partial correlation')
# plt.show()

## Save optimal model
with open(os.path.join(path, 'model.pkl'), 'wb') as output:
    dill.dump(varpc_final, output)

## Save results
np.savez_compressed(os.path.join(path, 'optimal_solution'),
                    lambda_A=lambda_A_opt,
                    lambda_rho=lambda_rho_opt,
                    A=A_final,
                    rho=rho_final,
                    c=c_final,
                    sparsity=sparsity_final,
                    fMSE_test=fMSE_test,
                    MSE_test=MSE_test,
                    fRsquared_test=fRsquared_test,
                    Rsquared_test=Rsquared_test)
