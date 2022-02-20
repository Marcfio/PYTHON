# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:09:01 2020

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

# Y5VGPLNQ2C2SYC0V7
key = '5VGPLNQ2C2SYC0V7'
ts = TimeSeries(key, output_format= 'pandas')



##################Market data##########################
data_apple_day, meta_data_day = ts.get_daily_adjusted(symbol='AAPL',outputsize='full')
(data_apple_day.columns)=['open_apple','high_apple','low_apple','apple_price','apple_adj','volume_apple','divid','split coef']
data_apple_price = data_apple_day.head(900)
data_apple_day.to_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\apple_data.csv')




plt.plot(data_apple_price.apple_price)


#####################Google trends data####################
kw_list_apple=["apple","iphone","apple watch","airpods"]

startday = datetime(2017,1,1)
endday = date.today()
apple_trends= msearch.dailydata2(kw_list_apple,startday,endday)
apple_trends.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\apple_trends_daily.csv")

plt.plot(apple_trends.index, apple_trends.iphone)
plt.plot
##### merge dataset ########

apple_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\apple_trends_daily.csv")
apple_trends=apple_trends.sort_values(by="date")
apple_trends.index= apple_trends.date


data_apple=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\apple_data.csv")
data_apple=data_apple.head(1400)
data_apple=data_apple.sort_values(by="date", ascending = False)
data_apple.index=data_apple.date


dataset = pd.concat([data_apple,apple_trends],axis=1, join='outer')
dataset=dataset[dataset.apple_price.notnull()]
dataset=dataset[dataset.apple.notnull()]
dataset = dataset.loc[:,~dataset.columns.duplicated()]
dataset.index=dataset.date
#dataset= dataset.drop(columns= dataset.columns[0])
dataset.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\dataset_apple.csv")


############import from csv file ###############################################################################


apple_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\apple_trends_daily.csv")
data_apple=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\apple_data.csv")
dataset=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\dataset_apple.csv")
#dataset = dataset.sort_values(by="date")
dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\dataset2_apple.csv")


#dataset2=dataset2.drop(dataset2.columns[[1]],axis='columns')



#############create dataset with dynamic correlation

dyncorr=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\dynamic_corr_apple.csv")
wei_dyn_corr =pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\dynamic_corr_wei_apple.csv")
apple_gen_trend= pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\gen_apple_trend.csv")

n_start_obs = len(dataset) - len(dyncorr)

dataset2=dataset.iloc[n_start_obs:]

dataset2['wei_dyn_corr']=wei_dyn_corr.values[:,1]
dataset2['dynamic_corr']=dyncorr.values[:,1]
dataset2['apple_gen_trend']=apple_gen_trend.values[n_start_obs:,1]



#dataset2 = dataset2.loc[:,~dataset2.columns.duplicated()]
#dataset2 = dataset2.drop('date.1',axis='columns')
#dataset2 = dataset2.drop(dataset2.columns[0],axis='columns')

dataset2.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\dataset2_apple.csv")

dataset2.columns=['Unnamed: 0', 'date', 'open_apple', 'high_apple', 'low_apple','apple_price', 'apple_adj', 'volume_apple', 'divid', 'split coef', 'apple', 'iphone', 'apple_watch', 'airpods', 'wei_dyn_corr',       'dynamic_corr', 'apple_gen_trend', 'wei_apple_trend']


dataset2['weighted_apple_trend'] = 0.65*dataset2.apple + 0.20*dataset2.iphone+ 0.075*dataset2.apple_watch +0.075*dataset2.airpods
dataset2['diff_wei_trend']=dataset2.wei_apple_trend.diff()

wei_appl_trend=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\wei_apple.csv")
wei_appl_trend.columns=['n','wei_apple_trend']
wei_appl_trend = wei_appl_trend.head(len(dataset2)- len(wei_appl_trend))

dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\dataset2_apple.csv")
dataset2['wei_apple_trend']=wei_appl_trend.wei_apple_trend

dataset2









############################################## import result##############################################à

result=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\result.csv")
result.columns=['date','position','cash','ret']
result['port_value']=result.position+ result.cash

####export dataset##############################################################
#result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\result_oil.csv")
#transactions.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\transaction_oil.csv")

#################################some plots#########################
#dataset2.plot(x = dataset2.columns[1] , y= ['tesla','elon','tesla_gen_trend'],title='google trends data', fontsize=8)


plt.figure()
plt.title('return of the strategy')
plt.plot(dataset2.date.values , result.ret)



plt.figure()
plt.title('portfolio value')
plt.plot(result.date , result.port_value)
#plt.plot( dataset2.tsla_price)




##############plot varible with different scale on y axis###à
fig = plt.figure('portfolio --- apple_price')
plt.title('portfolio --- apple_price')
ax1 = fig.add_subplot(111)
ax1.plot( dataset2.date,result.port_value)
ax1.set_ylabel('port_value $')

ax2 = ax1.twinx()
ax2.plot(dataset2.apple_price, 'r-')
ax2.set_ylabel('apple_price $', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')



##############correlation

port_apple_corr= np.corrcoef(result.port_value, dataset2.apple_price)
