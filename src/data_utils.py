from __future__ import division, print_function
import os

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import notebook
from IPython.display import Math
from datetime import date, timedelta
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def cosearch_fraction_directed(cosearch = None):
    row_sums = np.sum(cosearch, axis=1)
    indices = np.where(row_sums>0)[0]
    cosearchfrac = np.zeros(cosearch.shape,dtype=float)
    cosearchfrac[indices,:] = cosearch[indices,:]/np.sum(cosearch[indices,:], axis=1, keepdims=True)
    return cosearchfrac

def cosearchfraction_period(edgar_list=None,
                            load_dir=None,
                            N=None,
                            Years=None,
                            aggregation="overall-average"):
    files = []
    edgar_period = pd.DataFrame()
    cf_directed = []
    if aggregation=="overall-average":
        for year in Years:
            edgar_period = pd.concat([edgar_period, edgar_list[edgar_list['File'].str.contains('log{}'.format(year))]], axis=0)

        cosearch_directed = np.zeros((N,N), dtype=float)
        for link in edgar_period['File']:  
            filename = link.split('.zip')[0].split('log')[2]
            files.append(filename)

            # Load directed cosearch counts
            temp = sp.load_npz(os.path.join(load_dir,'cosearch'+filename+'.npz')).toarray()
            cosearch_directed += temp
        cf_directed.append(cosearch_fraction_directed(cosearch=cosearch_directed))
    elif aggregation=="annual-average":
        for year in Years:
            edgar_period = pd.concat([edgar_period, edgar_list[edgar_list['File'].str.contains('log{}'.format(year))]], axis=0)

            cosearch_directed = np.zeros((N,N), dtype=float)
            for link in edgar_period['File']:  
                filename = link.split('.zip')[0].split('log')[2]
                files.append(filename)

                # Load directed cosearch counts
                temp = sp.load_npz(os.path.join(load_dir,'cosearch'+filename+'.npz')).toarray()
                cosearch_directed += temp
            cf_directed.append(cosearch_fraction_directed(cosearch=cosearch_directed))
    return np.mean(cf_directed, axis=0)

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

def load_data(load_dir=None, 
              folder=None, 
              cosearch_ticker_file=None,
              residual_file=None,
              N_years=None):
    """Loads all time-series and returns the training, validation and test data alongwith cosearch graphs.
    
    Args:
        load_dir: data directory to load the time-series and cosearch files, str.
            - "/nfs/sloanlab003/projects/FinancialForecasting_proj/data/" on Sloan Engaging.
            - "/home/gridsan/shibal/data/"
        folder: folder is specific to the cohort used, str.
            - "EDGAR/Stanford-SP500"
            - "EDGAR/Stanford-SP1500"
        cosearch_ticker_file: name of ticker file, str.
            - "cik_ticker_sp500.csv"
            - "cik_ticker_sp1500.csv"
    Returns:
        df_train: training dataframe, dict of dataframes. 
        df_val: validation dataframe, dict of dataframes. 
        df_test: test dataframe, dict of dataframes. 
        cf_train: cosearch fraction for training period, a float numpy array of shape (N, N).
        cf_val: cosearch fraction for validation period, a float numpy array of shape (N, N).
        cf_test: cosearch fraction for test period, a float numpy array of shape (N, N). 
            [cf_test is ignored]
        df_companies: companies used in analysis, a numpy array of str.
    """
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

def load_all_timeseries_data(load_dir=None, 
                             folder=None,
                             cosearch_ticker_file=None,
                             training_years=np.arange(2005, 2015),
                             validation_years=np.arange(2015, 2016),
                             test_years=np.arange(2016, 2017),
                             aggregation="overall-average",
                             covariates=["LogAdjOpen", "LogAdjHigh", "LogAdjLow", "LogAdjClose",
                                         "pctAdjOpen", "pctAdjHigh", "pctAdjLow",
                                         "returns", "volume"],
                             response="volatilities"):
    """Loads all time-series and returns the training, validation and test data alongwith cosearch graphs.
    
    Args:
        load_dir: data directory to load the time-series and cosearch files, str.
            - "/nfs/sloanlab003/projects/FinancialForecasting_proj/data/" on Sloan Engaging.
            - "/home/gridsan/shibal/data/"
        folder: folder is specific to the cohort used, str.
            - "EDGAR/Stanford-SP500"
            - "EDGAR/Stanford-SP1500"
        cosearch_ticker_file: name of ticker file, str.
            - "cik_ticker_sp500.csv"
            - "cik_ticker_sp1500.csv"
        training_years: years for training period, int numpy array.
        validation_years: years for validation period, int numpy array.
        test_years: years for testing period, int numpy array.
        aggregation: aggregation method for generating cosearch fractions over the training/validation/test periods, str.
            - "overall-average", average of overall counts over the years.
            - "annual-average", average of annual fractions over the years.
        
        all_timeseries: Company specific features and response time-series, list of str.
        
    Returns:
        dfs_train: dictionary of dataframes per training time-series, dict of dataframes. 
        dfs_val: dictionary of dataframes per validation time-series, dict of dataframes. 
        dfs_test: dictionary of dataframes per test time-series, dict of dataframes. 
        cf_train: cosearch fraction for training period, a float numpy array of shape (N, N).
        cf_val: cosearch fraction for validation period, a float numpy array of shape (N, N).
        cf_test: cosearch fraction for test period, a float numpy array of shape (N, N). 
            [cf_test is ignored]
        df_companies: companies used in analysis, a numpy array of str.
    """
    all_timeseries = covariates
    all_timeseries.append(response)
    years = np.hstack([training_years, validation_years, test_years])
    years = np.sort(np.unique(years))
    # Finding submatrix cosearch fraction based on columns (Tickers) in returns/volatilities file   
    df_cosearch_ticker_file = pd.read_csv(os.path.join(load_dir, folder, cosearch_ticker_file), index_col=False)    
    ticker_to_index_mapping = df_cosearch_ticker_file['Ticker'].reset_index().set_index('Ticker').to_dict()['index']
            
    dfs = {} 
    for ts in all_timeseries:
        ### Read time series 
        filepath = os.path.join(load_dir, ts, "yahoo_companies_{}.csv".format(ts))
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        
        if ts in ["volume", "volatilities"]:
            df = df+1e-6
            df = df.apply(np.log10)
        
        ### Restrict to the time frame
        df = df[df.index.year.isin(years)]
        df = df.drop(df.index[0])
        
        ### Interpolate each time-series
        # df = df.interpolate(method='spline', order=3)
        df = df.interpolate(method='time')
        
        ### Drop companies which have NaNs even after interpolation
        df = df.dropna(axis=1)
        dfs[ts] = df
    
    # Find common tickers between cohort and the available time-series
    tickers = df_cosearch_ticker_file['Ticker'].values
    for key, df in dfs.items():
        tickers = np.intersect1d(tickers, df.columns.values)
    for key, df in dfs.items():
        dfs[key] = df[tickers]
        
        
        
    # Mapping between ticker and index to ensure knowledge graph index matches time-series columns
    indices_subset = np.array([int(i) for i in np.sort(dfs[list(dfs.keys())[0]].columns.map(ticker_to_index_mapping).dropna().values)])
    companies_subset = df_cosearch_ticker_file.loc[indices_subset,:]['Name'].values
    ticker_subset = df_cosearch_ticker_file.loc[indices_subset,:]['Ticker'].values
    ciks_subset = df_cosearch_ticker_file.loc[indices_subset,:]['CIK'].values
    
    indices = []
    companies = []
    tickers = []
    ciks = []
    for t, i, c, ck in zip(ticker_subset, indices_subset, companies_subset, ciks_subset):
        if t in df.columns:
            tickers.append(t)
            indices.append(i)
            companies.append(c)
            ciks.append(ck)
    
    for key, df in dfs.items():
#         df = df.interpolate(method='spline', order=3)
        df = df.interpolate(method='time')
        dfs[key] = df[tickers]
    
    
    edgar_list = pd.read_csv(os.path.join(load_dir, 'EDGAR', 'EDGAR-log-file-data-list.csv')) # Manually generated from 'https://www.sec.gov/files/EDGAR_LogFileData_thru_Jun2017.html'
    N = len(df_cosearch_ticker_file)

    ### Search-based peers identified using cosearch from previous year
    cf_train = cosearchfraction_period(
        edgar_list=edgar_list,
        load_dir=os.path.join(load_dir, folder, 'cosearchcounts'),
        N=N,
        Years=training_years[training_years>=2007],
        aggregation=aggregation
    )
    training_validation_years=np.hstack([training_years, validation_years])
    cf_val = cosearchfraction_period(
        edgar_list=edgar_list,
        load_dir=os.path.join(load_dir, folder, 'cosearchcounts'),
        N=N,
        Years=training_validation_years[training_validation_years>=2007],
        aggregation=aggregation
    )
    training_validation_test_years=np.hstack([training_years, validation_years, test_years])
    cf_test = cosearchfraction_period(
        edgar_list=edgar_list,
        load_dir=os.path.join(load_dir, folder, 'cosearchcounts'),
        N=N,
        Years=training_validation_test_years[training_validation_test_years>=2007],
        aggregation=aggregation
    )
    cf_train = cf_train[np.ix_(indices,indices)]
    cf_val = cf_val[np.ix_(indices,indices)]
    cf_test = cf_test[np.ix_(indices,indices)]
    cf_train = renormalize_cosearchfraction(cf_train)
    cf_val = renormalize_cosearchfraction(cf_val)
    cf_test = renormalize_cosearchfraction(cf_test)
    
    dfs_train = {}
    dfs_val = {}
    dfs_test = {}
    for key, df in dfs.items():
        dfs_train[key] = df[df.index.year.isin(training_years)]
        dfs_val[key] = df[df.index.year.isin(validation_years)]
        dfs_test[key] = df[df.index.year.isin(test_years)]
    
    
    df_companies = pd.DataFrame({'Ticker': tickers,
                                 'Company': companies,
                                 'CIK': ciks
                                })

    
    return dfs_train, dfs_val, dfs_test, cf_train, cf_val, cf_test, df_companies


def get_quarterly_report_component(tickers,
                                   component="BalanceSheet",
                                   load_directory="/home/gridsan/shibal/data/yahoo_finance",
                                   start_date=pd.datetime(2005,1,1),
                                   end_date=pd.datetime(2020,12,31)
                                  ):
    
    ########### Load Balance Sheets and Find Common features across tickers ########################
    dfs = dict()
    problematic_tickers = []
    deltaGap = relativedelta(months=6)
    for i, ticker in enumerate(tickers):
        try:
            if component=="BalanceSheet":
                df = pd.read_csv(os.path.join(load_directory, "Balance_Sheet_Quarter/{}_quarterly_balance-sheet.csv".format(ticker)), thousands=',').set_index('name').T
            elif component=="CashFlow":
                df = pd.read_csv(os.path.join(load_directory, "Cash_Flow_Quarter/{}_quarterly_cash-flow.csv".format(ticker)), thousands=',').drop(columns=['ttm']).set_index('name').T
            df.index.name='date'
            df.index = pd.to_datetime(df.index)
            df = df.loc[(df.index>=start_date-deltaGap)&(df.index<=end_date)]
            df.columns = df.columns.str.replace('\t', '')
            if i==0:
                common_features = set(df.columns)
            else:
                common_features = common_features&set(df.columns.values)
            dfs[ticker] = df.copy()
        except:
            problematic_tickers.append(ticker)
            continue
    
    print("Problematic Tickers: ", problematic_tickers)
    print("Common Features across companies in {}".format(component), common_features) 
    
    ########### Filter common features and resample at daily level ##################################
    for ticker, df in dfs.items():
        df = df[common_features]
#         df.index = df.index.to_period('Q')
        df = df.resample('D').ffill() 
        df = df.interpolate(method='time')
        df = df.loc[(df.index>=start_date)]
        dfs[ticker] = df
        
    ########### Generate dictionary of dataframes (feature: dataframe with all companies) ###########
    dfs_features = {}
    for feat in common_features:
        print(feat)
        for i, (ticker, df) in enumerate(dfs.items()):
            if i==0:
                df_feature = df[[feat]].copy()
                df_feature.columns = [ticker]
            else:
                df_feature[ticker] = df[feat]
        dfs_features[feat] = df_feature
#         display(dfs_features[feat])
    return dfs_features

def load_all_timeseries_data_with_dynamic_cosearch(load_dir=None, 
                             folder=None,
                             cosearch_ticker_file=None,
                             training_years=np.arange(2005, 2015),
                             validation_years=np.arange(2015, 2016),
                             test_years=np.arange(2016, 2017),
                             aggregation="overall-average",
                             covariates=["LogAdjOpen", "LogAdjHigh", "LogAdjLow", "LogAdjClose",
                                         "pctAdjOpen", "pctAdjHigh", "pctAdjLow",
                                         "returns", "volume"],
                             response="volatilities"):
    """Loads all time-series and returns the training, validation and test data alongwith cosearch graphs.
    
    Args:
        load_dir: data directory to load the time-series and cosearch files, str.
            - "/nfs/sloanlab003/projects/FinancialForecasting_proj/data/" on Sloan Engaging.
            - "/home/gridsan/shibal/data/"
        folder: folder is specific to the cohort used, str.
            - "EDGAR/Stanford-SP500"
            - "EDGAR/Stanford-SP1500"
        cosearch_ticker_file: name of ticker file, str.
            - "cik_ticker_sp500.csv"
            - "cik_ticker_sp1500.csv"
        training_years: years for training period, int numpy array.
        validation_years: years for validation period, int numpy array.
        test_years: years for testing period, int numpy array.
        aggregation: aggregation method for generating cosearch fractions over the training/validation/test periods, str.
            - "overall-average", average of overall counts over the years.
            - "annual-average", average of annual fractions over the years.
        
        all_timeseries: Company specific features and response time-series, list of str.
        
    Returns:
        dfs_train: dictionary of dataframes per training time-series, dict of dataframes. 
        dfs_val: dictionary of dataframes per validation time-series, dict of dataframes. 
        dfs_test: dictionary of dataframes per test time-series, dict of dataframes. 
        cf_train: cosearch fraction for training period, a float numpy array of shape (N, N).
        cf_val: cosearch fraction for validation period, a float numpy array of shape (N, N).
        cf_test: cosearch fraction for test period, a float numpy array of shape (N, N). 
            [cf_test is ignored]
        df_companies: companies used in analysis, a numpy array of str.
    """
    all_timeseries = covariates
    all_timeseries.append(response)
    years = np.hstack([training_years, validation_years, test_years])
    years = np.sort(np.unique(years))
    # Finding submatrix cosearch fraction based on columns (Tickers) in returns/volatilities file   
    df_cosearch_ticker_file = pd.read_csv(os.path.join(load_dir, folder, cosearch_ticker_file), index_col=False)    
    ticker_to_index_mapping = df_cosearch_ticker_file['Ticker'].reset_index().set_index('Ticker').to_dict()['index']
            
    dfs = {} 
    for ts in all_timeseries:
        ### Read time series 
        filepath = os.path.join(load_dir, ts, "yahoo_companies_{}.csv".format(ts))
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        
        if ts in ["volume", "volatilities"]:
            df = df+1e-6
            df = df.apply(np.log10)
        
        ### Restrict to the time frame
        df = df[df.index.year.isin(years)]
        df = df.drop(df.index[0])
        
        ### Interpolate each time-series
        # df = df.interpolate(method='spline', order=3)
        df = df.interpolate(method='time')
        
        ### Drop companies which have NaNs even after interpolation
        df = df.dropna(axis=1)
        dfs[ts] = df
    
    # Find common tickers between cohort and the available time-series
    tickers = df_cosearch_ticker_file['Ticker'].values
    for key, df in dfs.items():
        tickers = np.intersect1d(tickers, df.columns.values)
    for key, df in dfs.items():
        dfs[key] = df[tickers]
        
        
        
    # Mapping between ticker and index to ensure knowledge graph index matches time-series columns
    indices_subset = np.array([int(i) for i in np.sort(dfs[list(dfs.keys())[0]].columns.map(ticker_to_index_mapping).dropna().values)])
    companies_subset = df_cosearch_ticker_file.loc[indices_subset,:]['Name'].values
    ticker_subset = df_cosearch_ticker_file.loc[indices_subset,:]['Ticker'].values
    ciks_subset = df_cosearch_ticker_file.loc[indices_subset,:]['CIK'].values
    
    indices = []
    companies = []
    tickers = []
    ciks = []
    for t, i, c, ck in zip(ticker_subset, indices_subset, companies_subset, ciks_subset):
        if t in df.columns:
            tickers.append(t)
            indices.append(i)
            companies.append(c)
            ciks.append(ck)
    
    for key, df in dfs.items():
#         df = df.interpolate(method='spline', order=3)
        df = df.interpolate(method='time')
        dfs[key] = df[tickers]
    
    
    edgar_list = pd.read_csv(os.path.join(load_dir, 'EDGAR', 'EDGAR-log-file-data-list.csv')) # Manually generated from 'https://www.sec.gov/files/EDGAR_LogFileData_thru_Jun2017.html'
    N = len(df_cosearch_ticker_file)

    ### Search-based peers identified using cosearch from previous year
    cfs_train = {}
    for tr_y in training_years:
        if tr_y < 2007:
            tr_y_constrained = 2007
        else:
            tr_y_constrained = tr_y
        cfs_train[tr_y] = cosearchfraction_period(
                edgar_list=edgar_list,
                load_dir=os.path.join(load_dir, folder, 'cosearchcounts'),
                N=N,
                Years=np.array([tr_y_constrained]),
                aggregation=aggregation
            )
                         
    training_validation_years=np.hstack([training_years, validation_years])
    cf_val = cosearchfraction_period(
        edgar_list=edgar_list,
        load_dir=os.path.join(load_dir, folder, 'cosearchcounts'),
        N=N,
        Years=np.array([training_years[-1]]),
        aggregation=aggregation
    )
    training_validation_test_years=np.hstack([training_years, validation_years, test_years])
    cf_test = cosearchfraction_period(
        edgar_list=edgar_list,
        load_dir=os.path.join(load_dir, folder, 'cosearchcounts'),
        N=N,
        Years=np.array([validation_years[-1]]),
        aggregation=aggregation
    )
    cfs_train = {yr: cf_train[np.ix_(indices,indices)] for yr, cf_train in cfs_train.items()}
    cf_val = cf_val[np.ix_(indices,indices)]
    cf_test = cf_test[np.ix_(indices,indices)]
    cfs_train = {yr: renormalize_cosearchfraction(cf_train) for yr, cf_train in cfs_train.items()}
    cf_val = renormalize_cosearchfraction(cf_val)
    cf_test = renormalize_cosearchfraction(cf_test)
    
    dfs_train = {}
    dfs_val = {}
    dfs_test = {}
    for key, df in dfs.items():
        dfs_train[key] = df[df.index.year.isin(training_years)]
        dfs_val[key] = df[df.index.year.isin(validation_years)]
        dfs_test[key] = df[df.index.year.isin(test_years)]
    
    
    df_companies = pd.DataFrame({'Ticker': tickers,
                                 'Company': companies,
                                 'CIK': ciks
                                })

    
    return dfs_train, dfs_val, dfs_test, cfs_train, cf_val, cf_test, df_companies


def prepare_sequences(sequences, n_steps):
    """Prepares past-times samples as sequences.
    
    Args:
        sequences: a numpy array of shape (T, N)
        n_steps: num of past time steps to use as features/sequences, int scalar.
    
    Returns:
        X: past-time samples as features, a float numpy array of shape (T, p, N).
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
    return np.array(X), np.array(y)

def prepare_sequences_with_covariates(
        dfs,
        n_steps,
        covariates=['LogAdjOpen', 'LogAdjHigh', 'LogAdjLow', 'LogAdjClose', 'pctAdjOpen', 'pctAdjHigh', 'pctAdjLow', 'returns', 'volume'],
        response='volatilities'
    ):
    """Prepares past-times samples for covariates and response as sequences.
    
    Args:
        dfs: dictionary of pandas timeseries dataframes with shapes (T, N).
        n_steps: num of past time steps to use as features/sequences, int scalar.
    
    Returns:
        X: past-time samples for covariates and the target response as features, a float numpy array of shape (T, p, N, d).
        Y: target responses, a float numpy array of shape (T, N).
    """
    
    X = []
    for key in covariates:        
        x, _ = prepare_sequences(dfs.get(key).values, n_steps)
        X.append(x)
    
    x, Y = prepare_sequences(dfs.get(response).values, n_steps)
    X.append(x)
    X = np.array(X)
    X = np.transpose(X, axes=[1,2,3,0])    
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

def prepare_grocery_data(load_dir='/home/gridsan/shibal/FinancialForecasting_proj/data/',
                         folder='SpatioTemporal_data/grocery-sales-forecasting',
                         num_items=3000):
    """Prepares train, val, test and graph.
    Args:
        df: timeseries dataframe with col: date, store_nbr, item_nbr unit_sales and promotion

    Returns:
        dfs_train
        dfs_val
        dfs_test
    """
    df_dir = os.path.join(load_dir, folder)
    train_file = os.path.join(df_dir, "train.csv")
    df_train_total = pd.read_csv(train_file,
    #                      converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
                     parse_dates=["date"])
    df_train_total = df_train_total.loc[df_train_total.date!=pd.datetime(2014,1,1)] # New Years holiday
    df_train_total = df_train_total.loc[df_train_total.date!=pd.datetime(2015,1,1)] # New Years holiday
    df_train_total = df_train_total.loc[df_train_total.date!=pd.datetime(2016,1,1)] # New Years holiday
    df_train_total = df_train_total.loc[df_train_total.date!=pd.datetime(2017,1,1)] # New Years holiday
    df_train_withPromo = df_train_total[21657651:]  ###select date with promotion data available

    df_all = df_train_withPromo.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0) 
    df_all.columns = df_all.columns.get_level_values(1)
    df_all = df_all.T
    for col in df_all.columns:
        df_all[col] = df_all[col].apply(lambda u: np.log1p(u) if u > 0 else 0)
    df_all = df_all.diff()
    df_all = df_all.T
    df_all = df_all.reset_index().groupby(['item_nbr'])[df_all.columns].mean().dropna(axis=1) # average across stores

    items_selected = (df_all!=0).sum(axis=1).sort_values(ascending=False).index[:num_items]

    df_train = df_train_withPromo.loc[df_train_withPromo.date<pd.datetime(2016,4,1)]
    df_val = df_train_withPromo.loc[(df_train_withPromo.date>=pd.datetime(2016,4,1))&(df_train_withPromo.date<pd.datetime(2017,1,1))]
    df_test = df_train_withPromo.loc[df_train_withPromo.date>=pd.datetime(2017,1,1)]
    df_store_item_sales = df_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
    item_file = os.path.join(df_dir, "items.csv")
    items = pd.read_csv(item_file).set_index("item_nbr")
    items = items[items.index.isin(items_selected)]

    classes_selected = np.intersect1d(
        np.intersect1d(
        df_train['item_nbr'].map(dict(pd.DataFrame(items['class']).reset_index().values)).dropna().unique(),
        df_val['item_nbr'].map(dict(pd.DataFrame(items['class']).reset_index().values)).dropna().unique()
        ),
        df_test['item_nbr'].map(dict(pd.DataFrame(items['class']).reset_index().values)).dropna().unique()
    )
    items = items[items['class'].isin(classes_selected)]
    items_selected = items.index.unique()

    #     class_itemCount = pd.DataFrame(items.reset_index().groupby('class')['item_nbr'].size()).reset_index()     
    #     class_family_pair = items[items['class'].isin(dfs[0]['class_sales'].columns)].sort_values(by=['family', 'class']).set_index('class')['family'].reset_index().drop_duplicates() ##for sorting
    class_family_pair = items.sort_values(by=['family', 'class']).set_index('class')['family'].reset_index().drop_duplicates() ##for sorting
    #     items_tmp = items.reindex(df_store_item_sales.index.get_level_values(1))
    #     items_train = items.loc[items['class'].isin(items_tmp['class'].unique())]
    train = True
    sales_scaler = StandardScaler()
    promotion_scaler = StandardScaler()
    countStore_scaler = StandardScaler()
    dfs = []
    for df in [df_train, df_val, df_test]:
        df = df[df["item_nbr"].isin(items_selected)]
        df_store_item_promo = df.reset_index().set_index(
        ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
            level=-1).fillna(False)
        df_store_item_promo.columns = df_store_item_promo.columns.get_level_values(1)
        df_store_item_sales = df.set_index(
        ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
            level=-1).fillna(0)
        df_store_item_sales.columns = df_store_item_sales.columns.get_level_values(1)
    #         items2 = items_train.reindex(df_store_item_sales.index.get_level_values(1)) ###only get the item appeared in training data
        df_promo_store_class = df_store_item_promo.reset_index()
        df_promo_store_class['class'] = df_promo_store_class['item_nbr'].map(dict(pd.DataFrame(items['class']).reset_index().values))
    #         df_promo_store_class_index = df_promo_store_class[['class', 'store_nbr']]
        df_store_class = df_store_item_sales.reset_index()
        df_store_class['class'] = df_store_class['item_nbr'].map(dict(pd.DataFrame(items['class']).reset_index().values))
    #         df_store_class_index = df_store_class[['class', 'store_nbr']]
        df_store_class = df_store_class.groupby(['class', 'store_nbr'])[df_store_item_sales.columns].sum()

        df_store_item = df_store_item_sales.reset_index()
        df_store_item = df_store_item.set_index(['store_nbr','item_nbr'])[df_store_item_sales.columns].T
        # Ignore returned items! Take log transformation
        for col in df_store_item.columns:
            df_store_item[col] = df_store_item[col].apply(lambda u: np.log1p(u) if u > 0 else 0)  
        # Take difference to compute log(s_t/s_{t-1})
        df_store_item = df_store_item.diff()
        df_store_item = df_store_item.T
        df_item_sales = df_store_item.reset_index().groupby(['item_nbr'])[df_store_item_sales.columns].mean() # average across stores
        df_item_sales = df_item_sales.reset_index()
        df_item_sales['class'] = df_item_sales['item_nbr'].map(dict(pd.DataFrame(items['class']).reset_index().values))
        df_class_sales = df_item_sales.groupby(['class'])[df_store_item_sales.columns].mean().dropna(axis=1) # average across items in a class
        df_class_sales_family = df_class_sales.reset_index().merge(class_family_pair, on='class', how='inner')
        df_class_sales_sorted = df_class_sales_family.sort_values(by=['family', 'class']).set_index('class').drop(columns='family')

        df_class_promotion = df_promo_store_class.groupby(['class', 'item_nbr'])[df_store_item_promo.columns].mean() # average across stores
        df_class_promotion = df_class_promotion.reset_index().groupby(['class'])[df_store_item_sales.columns].mean() # average across items       
    #         df_class_promotion = df_promo_store_class.reset_index().groupby(['class'])[df_store_item_sales.columns].sum() ###daily number of promotion
    #         df_class_promotion_itemCount = df_class_promotion.reset_index().merge(class_itemCount, on='class', how='left')
    #         df_class_promotion_itemCount = df_class_promotion_itemCount.set_index(['class'])
    #         df_class_promotion = df_class_promotion_itemCount.iloc[:,:-1].div(df_class_promotion_itemCount["item_nbr"], axis=0)
        df_class_promotion_family = df_class_promotion.reset_index().merge(class_family_pair, on='class', how='inner')
        df_class_promotion_sorted = df_class_promotion_family.sort_values(by=['family', 'class']).set_index('class').drop(columns='family')        
        
        
        df_class_countStore = df_store_class.reset_index().groupby('class')[df_store_item_sales.columns].agg(lambda x: x.ne(0).sum())/len(df_store_class.reset_index()['store_nbr'].unique()) ###fraction of stores selling the items in the class
        df_class_countStore_family = df_class_countStore.reset_index().merge(class_family_pair, on='class', how='inner')
        df_class_countStore_sorted = df_class_countStore_family.sort_values(by=['family', 'class']).set_index('class').drop(columns='family')

        if train:
            sales_scaler.fit(df_class_sales_sorted.T)
            promotion_scaler.fit(df_class_promotion_sorted.T)
            countStore_scaler.fit(df_class_countStore_sorted.T)
            train = False

            
        class_sales_scale = sales_scaler.transform(df_class_sales_sorted.T)
        df_class_sales_scale = pd.DataFrame(class_sales_scale)
        df_class_sales_scale.columns = df_class_sales_sorted.T.columns.astype(int)
        df_class_sales_scale.index = df_class_sales_sorted.T.index
#         df_class_sales_scale = df_class_sales_scale.clip(-5, 5)
        
        df_class_promotion_scale = df_class_promotion_sorted.T
        df_class_promotion_scale = df_class_promotion_scale[df_class_promotion_scale.index.isin(df_class_sales_scale.index)]
        df_class_promotion_scale.columns = df_class_promotion_scale.columns.astype(int)
#         class_promotion_scale = promotion_scaler.transform(df_class_promotion_sorted.T)
#         df_class_promotion_scale = pd.DataFrame(class_promotion_scale)
#         df_class_promotion_scale.columns = df_class_promotion_sorted.T.columns.astype(int)
#         df_class_promotion_scale.index = df_class_promotion_sorted.T.index
#         df_class_promotion_scale = df_class_promotion_scale[df_class_promotion_scale.index.isin(df_class_sales_scale.index)]

        df_class_countStore_scale = df_class_countStore_sorted.T
        df_class_countStore_scale = df_class_countStore_scale[df_class_countStore_scale.index.isin(df_class_sales_scale.index)]
        df_class_countStore_scale.columns = df_class_countStore_scale.columns.astype(int)
#         class_countStore_scale = countStore_scaler.transform(df_class_countStore_sorted.T)
#         df_class_countStore_scale = pd.DataFrame(class_countStore_scale)
#         df_class_countStore_scale.columns = df_class_countStore_sorted.T.columns.astype(int)
#         df_class_countStore_scale.index = df_class_countStore_sorted.T.index
#         df_class_countStore_scale = df_class_countStore_scale[df_class_countStore_scale.index.isin(df_class_sales_scale.index)]
        
        dfs.append(
            {'class_promotion':df_class_promotion_scale[df_class_sales_scale.columns],
             'class_countStore':df_class_countStore_scale[df_class_sales_scale.columns],
             'class_sales':df_class_sales_scale.clip(-10, 10),
             }
        )
    dfs_train, dfs_val, dfs_test = dfs
            
    class_family = {}
    for i in range(len(class_family_pair)):
        class_family[class_family_pair.iloc[i,:]['class']]=class_family_pair.iloc[i,:]['family']
    return dfs_train, dfs_val, dfs_test, class_family

def prepare_grocery_data_old(load_dir='/home/gridsan/shibal/FinancialForecasting_proj/data/',
                             folder='SpatioTemporal_data/grocery-sales-forecasting',
                             difference=False):
    """Prepares train, val, test and graph.
    Args:
        df: timeseries dataframe with col: date, store_nbr, item_nbr unit_sales and promotion

    Returns:
        dfs_train
        dfs_val
        dfs_test
    """
    df_dir = os.path.join(load_dir, folder)
    train_file = os.path.join(df_dir, "train.csv")
    df_train_total = pd.read_csv(train_file,
                     converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
                     parse_dates=["date"])
    df_train_withPromo = df_train_total[21657651:]  ###select date with promotion data available
    df_train = df_train_withPromo.loc[df_train_withPromo.date<pd.datetime(2016,6,1)]
    df_val = df_train_withPromo.loc[(df_train_withPromo.date>=pd.datetime(2016,6,1))&(df_train_withPromo.date<pd.datetime(2017,1,1))]
    df_test = df_train_withPromo.loc[df_train_withPromo.date>=pd.datetime(2017,1,1)]
    df_store_item_sales = df_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
    item_file = os.path.join(df_dir, "items.csv")
    items = pd.read_csv(item_file).set_index("item_nbr")
    class_itemCount = pd.DataFrame(items.reset_index().groupby('class')['item_nbr'].size()).reset_index()
    class_family_pair = pd.DataFrame(items.reset_index().groupby('class')['family'].max()).reset_index() ##for sorting
    items_tmp = items.reindex(df_store_item_sales.index.get_level_values(1))
    items_train = items.loc[items['class'].isin(items_tmp['class'].unique())]
    le = LabelEncoder()
    items['family'] = le.fit_transform(items['family'].values)
    train = True
    sales_scaler = StandardScaler()
    promotion_scaler = StandardScaler()
    countStore_scaler = StandardScaler()
    dfs = []
    for df in [df_train, df_val, df_test]:
        df_store_item_promo = df.reset_index().set_index(
        ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
            level=-1).fillna(False)
        df_store_item_promo.columns = df_store_item_promo.columns.get_level_values(1)
        df_store_item_sales = df.set_index(
        ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
            level=-1).fillna(0)
        df_store_item_sales.columns = df_store_item_sales.columns.get_level_values(1)
        items2 = items_train.reindex(df_store_item_sales.index.get_level_values(1)) ###only get the item appeared in training data
        df_promo_store_class = df_store_item_promo.reset_index()
        df_promo_store_class['class'] = items2['class'].values
        df_promo_store_class_index = df_promo_store_class[['class', 'store_nbr']]
        df_promo_store_class = df_promo_store_class.groupby(['class', 'store_nbr'])[df_store_item_promo.columns].sum()
        df_store_class = df_store_item_sales.reset_index()
        df_store_class['class'] = items2['class'].values
        df_store_class_index = df_store_class[['class', 'store_nbr']]
        df_store_class = df_store_class.groupby(['class', 'store_nbr'])[df_store_item_sales.columns].sum()
        
        df_class_sales = df_store_class.reset_index().groupby(['class'])[df_store_item_sales.columns].sum() ###sales history
        df_class_sales_family = df_class_sales.reset_index().merge(class_family_pair, on='class', how='inner')
        df_class_sales_sorted = df_class_sales_family.sort_values('family').set_index('class').drop(columns='family')

        df_class_promotion = df_promo_store_class.reset_index().groupby(['class'])[df_store_item_sales.columns].sum() ###daily number of promotion
        df_class_promotion_itemCount = df_class_promotion.reset_index().merge(class_itemCount, on='class', how='left')
        df_class_promotion_itemCount = df_class_promotion_itemCount.set_index(['class'])
        df_class_promotion = df_class_promotion_itemCount.iloc[:,:-1].div(df_class_promotion_itemCount["item_nbr"], axis=0)
        df_class_promotion_family = df_class_promotion.reset_index().merge(class_family_pair, on='class', how='inner')
        df_class_promotion_sorted = df_class_promotion_family.sort_values('family').set_index('class').drop(columns='family')        
        
        df_class_countStore = df_store_class.reset_index().groupby('class')[df_store_item_sales.columns].agg(lambda x: x.ne(0).sum()) ###daily number of stores selling the items in the class
        df_class_countStore_family = df_class_countStore.reset_index().merge(class_family_pair, on='class', how='inner')
        df_class_countStore_sorted = df_class_countStore_family.sort_values('family').set_index('class').drop(columns='family')
        
        if train:
            sales_scaler.fit(df_class_sales_sorted.T)
            promotion_scaler.fit(df_class_promotion_sorted.T)
            countStore_scaler.fit(df_class_countStore_sorted.T)
            train = False

        class_sales_scale = sales_scaler.transform(df_class_sales_sorted.T)
        df_class_sales_scale = pd.DataFrame(class_sales_scale)
        df_class_sales_scale.columns = df_class_sales_sorted.T.columns.astype(int)
        
        class_promotion_scale = promotion_scaler.transform(df_class_promotion_sorted.T)
        df_class_promotion_scale = pd.DataFrame(class_promotion_scale)
        df_class_promotion_scale.columns = df_class_promotion_sorted.T.columns.astype(int)
        
        class_countStore_scale = countStore_scaler.transform(df_class_countStore_sorted.T)
        df_class_countStore_scale = pd.DataFrame(class_countStore_scale)
        df_class_countStore_scale.columns = df_class_countStore_sorted.T.columns.astype(int)
        
        dfs.append(
            {'class_promotion':df_class_promotion_scale[df_class_sales_scale.columns],
             'class_countStore':df_class_countStore_scale[df_class_sales_scale.columns],
             'class_sales':df_class_sales_scale,
             }
        )
               
    dfs_train, dfs_val, dfs_test = dfs
            
    class_family = {}
    for i in range(len(class_family_pair)):
        class_family[class_family_pair.loc[i,'class']]=class_family_pair.loc[i,'family']
    return dfs_train, dfs_val, dfs_test, class_family

def prepare_currency_data(load_dir='/home/gridsan/shibal/FinancialForecasting_proj/data/',
                          folder='currency',
                          start_date=pd.datetime(2005,1,1),
                          end_date=pd.datetime(2020,12,31)):
    df_dir = os.path.join(load_dir, folder)
    file = os.path.join(df_dir, "currencyRates_184_1999_2021.csv")
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df[(df.index>=start_date)&(df.index<=end_date)].dropna(axis=1)
    df = df.apply(np.log)
    df = df.diff().dropna(axis=0)                                                         

    currencies_selected = (df<1e-7).sum().sort_values()[:100].index
    df = df[currencies_selected]

    df_train = df[(df.index>=pd.datetime(2005,1,1))&(df.index<=pd.datetime(2016,12,31))]
    df_val = df[(df.index>=pd.datetime(2017,1,1))&(df.index<=pd.datetime(2018,12,31))]
    df_test = df[(df.index>=pd.datetime(2019,1,1))&(df.index<=pd.datetime(2020,12,31))]                                           
                                                         
    scaler = StandardScaler()
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
    df_train_scaled.index = df_train.index

    df_val_scaled = pd.DataFrame(scaler.transform(df_val), columns=df_val.columns)
    df_val_scaled.index = df_val.index

    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)
    df_test_scaled.index = df_test.index
    
#     df_train_scaled = df_train_scaled.clip(-10, 10)
#     df_val_scaled = df_val_scaled.clip(-10, 10)
#     df_test_scaled = df_test_scaled.clip(-10, 10)
                                                         
    return {'currency': df_train_scaled}, {'currency': df_val_scaled}, {'currency': df_test_scaled}
                                                         
# def compute_metrics(X=None,
#                     Y=None, 
#                     f=None, 
#                     rho=None,
#                     c=None):
#     """Computes Evaluation metrics.
    
#     Args:
#         X: past-time samples as features, a float numpy array of shape (T, N, p).
#         Y: target responses, a float numpy array of shape (T, N).
#         f: forecast responses, a float numpy array of shape (T, N).
#         rho: partial correlation parameters (symmetric), a float numpy array of shape (N, N).
#         c: inverse of conditional variances, a float numpy array of shape (N, ).
    
#     Returns:
#         mse: mean squared error, float scalar.
#         rsquared: correlation R^2, float scalar.
#         frsquared: forecasting R^2, float scalar.
        
#         Note we use the definition of R^2 in "Nets:  Network esti-mation  for  time  series" by M. Barigozzi and C. Brownlees.  
#     """
#     T, N = Y.shape
#     eps = Y - f
#     Theta = rho * np.sqrt(c / c[:, None])
#     Ypred = f + eps@Theta.T
        
#     fmse = mean_squared_error(Y.reshape(-1), f.reshape(-1), multioutput='uniform_average') 
#     mse = mean_squared_error(Y.reshape(-1), Ypred.reshape(-1), multioutput='uniform_average') 
#     f_rsquared = np.mean([1-mean_squared_error(Y[:,j], f[:,j])/mean_squared_error(Y[:,j], np.zeros(T)) for j in range(N)])
#     rsquared = np.mean([1-mean_squared_error(Y[:,j], Ypred[:,j])/mean_squared_error(Y[:,j], np.zeros(T)) for j in range(N)])
#     return fmse, mse, f_rsquared, rsquared
