import pandas as pd
import numpy as np
import ta 
import shap
import pickle
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV, train_test_split

PATH = './data/'

# Get Data 
def get_data(PATH, stock_filename, sp500_filename):
    all_stocks = pd.read_csv(PATH + stock_filename)
    sector = pd.read_csv(PATH + sp500_filename)
    all_stocks  = all_stocks.merge(sector[['Symbol', 'Sector']], how='left', on='Symbol')
    return all_stocks 

def add_sector(all_stocks): 
    # update unknown sector 
    all_stocks.loc[all_stocks['Symbol']=='CEG','Sector']='Utilities'
    all_stocks.loc[all_stocks['Symbol']=='ELV','Sector']='Healthcare'
    all_stocks.loc[all_stocks['Symbol']=='GEN','Sector']='Technology'
    all_stocks.loc[all_stocks['Symbol']=='META','Sector']='Communication Services'
    all_stocks.loc[all_stocks['Symbol']=='PARA','Sector']='Communication Services'
    all_stocks.loc[all_stocks['Symbol']=='SBUX','Sector']='Consumer Cyclical'
    all_stocks.loc[all_stocks['Symbol']=='V','Sector']='Financial Services'
    all_stocks.loc[all_stocks['Symbol']=='WBD','Sector']='Communication Services'
    all_stocks.loc[all_stocks['Symbol']=='WTW','Sector']='Financial Services'
    return all_stocks

def calculate_returns(all_stocks):
    # calculate return as a log difference
    all_stocks = all_stocks.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    all_stocks['adj_close_lag1'] = all_stocks[['Symbol', 'Date', 'Adj Close']].groupby(['Symbol']).shift(1)['Adj Close'].reset_index(drop=True)
    all_stocks['return'] = np.log(all_stocks['Adj Close']/all_stocks['adj_close_lag1'])
    return all_stocks

def create_lagged_features(df, var):
    df[var + '_lag1'] = df[['Symbol', 'Date', var]].groupby(['Symbol']).shift(1)[var].reset_index(drop=True)
    df[var+'_rolling5'] = df[['Symbol', 'Date', var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(5).sum().reset_index(drop=True)
    df[var+'_rolling15'] = df[['Symbol', 'Date', var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(15).sum().reset_index(drop=True)
    return df

def implement_lags(all_stocks):
    df = create_lagged_features(all_stocks, 'return')
    df = create_lagged_features(all_stocks, 'Volume')
    df['relative_vol_1_15'] = df['Volume_lag1']/df['Volume_rolling15']
    df['relative_vol_5_15'] = df['Volume_rolling5']/df['Volume_rolling15']
    return df

def transform_sector(all_stocks):
    sector_counts = all_stocks['Sector'].value_counts()
    enc = OrdinalEncoder(categories=[list(sector_counts.index)])
    all_stocks['Sector_enc'] = enc.fit_transform(all_stocks[['Sector']])
    return all_stocks

def add_exponential_smoothing(all_stocks):
    all_stocks['ema50'] = all_stocks['Adj Close']/all_stocks['Adj Close'].ewm(50).mean()
    all_stocks['ema21'] = all_stocks['Adj Close']/all_stocks['Adj Close'].ewm(21).mean()
    all_stocks['ema15'] = all_stocks['Adj Close']/all_stocks['Adj Close'].ewm(15).mean()
    all_stocks['ema5'] = all_stocks['Adj Close']/all_stocks['Adj Close'].ewm(5).mean()
    return all_stocks

def normalize_vol(all_stocks):
    all_stocks['normVol'] = all_stocks['Volume'] / all_stocks['Volume'].ewm(5).mean()
    return all_stocks

def create_zscores(all_stocks):
    zscore_fxn = lambda x: (x - x.mean())/ x.std()
    all_stocks['zscore'] = all_stocks.groupby('Symbol', group_keys=False)['Adj Close'].apply(zscore_fxn)
    return all_stocks

def add_technical_analysis(all_stocks):
    all_stocks['ta'] = ta.volume.money_flow_index(all_stocks.High, all_stocks.Low, all_stocks.Close,
        all_stocks.Volume, window=14, fillna=False)
    # mean center the data
    all_stocks['ta_mean_centered'] = all_stocks['ta'] - all_stocks['ta'].rolling(50, min_periods=20).mean()
    return all_stocks

def get_shap_values(all_stocks, lag):
    list_shap_vals = []
    all_stocks.columns = [str(x).lower().replace(' ', '_') for x in all_stocks.columns]
    this_stock ='WMT'
    feature_list = ['sector_enc', 'return_lag1', 'return_rolling5', 'return_rolling15', 'relative_vol_1_15',
        'ema50', 'ema21', 'ema15', 'ema5', 'normvol', 'zscore', 'ta_mean_centered']
    this_date = all_stocks.loc[all_stocks.index[-lag], 'date']
    today_stocks = all_stocks[np.logical_and(all_stocks['date']==this_date, all_stocks['symbol'] !=this_stock)]
    
    X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], 
        test_size=0.1, random_state=42)
    param_grid = {'max_depth': list(range(3, 7, 1))}
    params_fit = {'eval_metric': "mae", 'eval_set': [[X_test, y_test]], 'early_stopping_rounds':10}
    gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.75, gamma=0, learning_rate=0.05, 
        max_delta_step=0, missing=-99999, n_estimators=300, random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
        seed=None, silent=None, subsample=.5, verbosity=1)
    search = GridSearchCV(gbm, param_grid=param_grid)
    search.fit(X_train, y_train, **params_fit)

    this_data = all_stocks[np.logical_and(all_stocks['date']==this_date, all_stocks['symbol']==this_stock)][feature_list]
    this_actual = all_stocks[np.logical_and(all_stocks['date']==this_date, all_stocks['symbol']==this_stock)]['return']
    search.best_estimator_.predict(this_data), this_actual
    explainer = shap.TreeExplainer(search.best_estimator_)
    shap_values = explainer.shap_values(this_data)
    temp = pd.DataFrame(shap_values[0], index=this_data.columns, columns=['day' + str(lag)])
    temp.loc['base'] = explainer.expected_value
    list_shap_vals.append(temp)    
    return pd.concat(list_shap_vals)

def combine_shap_values(all_stocks):
    output = pd.DataFrame()
    for i in range(3, 13):
        output['day' + str(i)] = get_shap_values(all_stocks, i)
    output = output.transpose()
    return output.reset_index(drop=False)

def save_shap_values(shaps):
    with open(PATH + 'shaps.pickle', 'wb') as f:
        pickle.dump(shaps, f)

def create_stacked_bar_chart():
    output_transposed.plot(x='index', kind='bar', stacked=True,
        title="SHAP Values for 10 Day Period")



if __name__ == "__main__":
    data = get_data(PATH, "sp500_stocks.csv", "sp500_companies.csv")
    all_stocks = add_sector(data)
    all_stocks = calculate_returns(all_stocks)
    all_stocks = implement_lags(all_stocks)
    all_stocks = transform_sector(all_stocks)
    all_stocks = add_exponential_smoothing(all_stocks)
    all_stocks = normalize_vol(all_stocks)
    all_stocks = create_zscores(all_stocks)
    all_stocks = add_technical_analysis(all_stocks)
    shaps = combine_shap_values(all_stocks)
    save_shap_values(shaps)
    # print(shaps)
    # print(all_stocks.iloc[0]) #tail(n=20))