# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:18:34 2020

@author: MARCOFIORAVANTIPC
"""

import msearch
import pickle
import pandas as pd
from datetime import datetime
from datetime import date
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR ,  SVAR
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from pytrends import dailydata
from alpha_vantage.timeseries import TimeSeries
import statsmodels.api as sm
from statsmodels.tsa.api import VAR ,  SVAR
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# Y5VGPLNQ2C2SYC0V7
key = '5VGPLNQ2C2SYC0V7'
ts = TimeSeries(key, output_format= 'pandas')



####################alphalens daily data#########################

data_spx_day, meta_data_day = ts.get_daily_adjusted(symbol='SPX',outputsize='full')
(data_spx_day.columns)=['open_spx','high_spx','low_spx','sp500','sp500_adj','volume_spx','divid','split coef']
data_spx_day.to_csv(r'S&P500.csv')

data_spx = data_spx_day.head(900)
spx_ret = data_spx.pct_change()
spx_ret.index = pd.DatetimeIndex(spx_ret.index)
spx_ret = spx_ret.sort_index()

################# from google trends ############################à
kw_list_trump=["Trump"]
kw_list_gen= ["crisis","USA",'Chinese']
startday = datetime(2017,1,1)
endday = date.today()
trump_trends= msearch.dailydata2(kw_list_trump,startday,endday)
#trump_trends.to_csv("trump_trends_daily.csv")
gen_trends= msearch.dailydata2(kw_list_gen,startday,endday)
trend_data=trump_trends.merge(gen_trends)
trends_ret= trump_trends.pct_change()  ####diff data
trends_ret.to_csv("trends_ret.csv")
trends_ret.index=pd.DatetimeIndex(trends_ret.index)
trends_ret=trends_ret.sort_index()
trend = pd.concat([gen_trends,trump_trends],axis=1, join='outer')

##########################import data from csv#####################
dataset=pd.read_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dataset.csv')
dataset['date']=pd.DatetimeIndex(dataset.date)

# sep=pd.read_csv('S&P500.csv')
# sep=sep.sort_index( axis=0,  ascending=False, inplace=False )
# sep.to_csv('S&P500.csv')


dataset_ret =pd.read_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dataset_ret.csv')
dataset.index = pd.DatetimeIndex.strftime(dataset.date,'%Y-%m-%d %H:%M:%S')

dataset= dataset.drop(columns= 'date')
dataset.to_csv('dataset.csv')

din_corr=pd.read_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dynamic_corr.csv')
dataset['dynamic_corr']=din_corr.values[:,1]

##################################import data for the dynamic correlation 2 ---> for loop ################
dataset=pd.read_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dataset.csv')
dyncorr2 = pd.read_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dynamic_corr2.csv')
n_start_obs = len(dataset) - len(dyncorr2)

dataset2=dataset.iloc[n_start_obs:]
dataset2['dynamic_corr2']=dyncorr2.values[:,1]

plt.figure()
dataset2.plot(x='date', y= ['dynamic_corr','dynamic_corr2'])
dataset2.to_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dataset2.csv')

##########################################merge dataset########

dataset = pd.concat([data_spx,trend],axis=1, join='outer')
dataset= dataset[dataset.Trump.notnull()]
dataset= dataset[dataset.sp500.notnull()]

dataset_ret = pd.concat([spx_ret,trends_ret],axis=1, join='outer')
dataset_ret= dataset_ret[dataset_ret.Trump.notnull()]
dataset_ret= dataset_ret[dataset_ret.sp500.notnull()]


dataset_ret.to_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dataset_ret.csv')
dataset.index=dataset.date
dataset=dataset.drop(dataset.date)
dataset.to_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dataset.csv')

#######################static correlation analysis############
corr = dataset.corr()
corr_diff = dataset_ret.corr()


