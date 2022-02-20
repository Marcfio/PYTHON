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
from alpha_vantage.foreignexchange import ForeignExchange as fe
# Y5VGPLNQ2C2SYC0V7
key = '5VGPLNQ2C2SYC0V7'
ts = TimeSeries(key, output_format= 'pandas')
fx = fe(key, output_format='pandas')


##################Market data##########################
data_dollar_day , meta_data = fx.get_currency_exchange_daily(from_symbol="USD", to_symbol = "EUR",  outputsize = 'full' )
(data_dollar_day.columns)=['open_dollar_eur','high_dollar_eur','low_dollar_eur','dollar_eur']
data_dollar_eur = data_dollar_day.head(900)

data_dollar_day.to_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dollar_eur_data.csv')




plt.plot(data_dollar_eur)


#####################Google trends data####################
kw_list_dollar=["crisis","job","loan","unemployment","default", "suicide", "killing" , "disorder" , "crime" , "usurer" , "war" , "uprising"]
kw_list_unemploy=["hiring","work","usajob","apply"]
startday = datetime(2017,1,1)
endday = date.today()
dollar_trends= msearch.dailydata2(kw_list_dollar,startday,endday)
dollar_trends.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dollar_trends_daily.csv")

employment_trends=msearch.dailydata2(kw_list_unemploy,startday,endday)
employment_trends.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\employment_trends.csv")

dollar_trends['hiring']=employment_trends.hiring.values[0:-8]
dollar_trends['work']=employment_trends.work.values[0:-8]
dollar_trends['usajob']=employment_trends.usajob.values[0:-8]
dollar_trends['apply']=employment_trends.values[0:-8,3]



plt.plot(dollar_trends.index, dollar_trends.iphone)
##### merge dataset ########

dollar_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dollar_trends_daily.csv")
dollar_trends=dollar_trends.sort_values(by="date")
dollar_trends.index= dollar_trends.date


data_dollar_eur=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dollar_eur_data.csv")
data_dollar_eur=data_dollar_eur.head(900)
data_dollar_eur=data_dollar_eur.sort_values(by="date", ascending = True)
data_dollar_eur.index=data_dollar_eur.date


dataset = pd.concat([data_dollar_eur,dollar_trends],axis=1, join='outer')
dataset=dataset[dataset.dollar_eur.notnull()]
dataset=dataset[dataset.crisis.notnull()]
dataset = dataset.loc[:,~dataset.columns.duplicated()]
dataset.index=dataset.date
#dataset2= dataset2.drop(columns= dataset2.columns[0])
dataset.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dataset_dollar_eur.csv")


############import from csv file ###############################################################################


dollar_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dollar_trends_daily.csv")
data_dollar_eur=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dollar_eur_data.csv")
dataset=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dataset_dollar_eur.csv")
#dataset = dataset.sort_values(by="date")
dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dataset2_dollar_eur.csv")


#dataset2=dataset2.drop(dataset2.columns[[1]],axis='columns')

dollar_trends = pd.concat([dollar_trends,employment_trends],axis=1, join='outer')
dataset=dataset[dataset.dollar_eur.notnull()]
dataset=dataset[dataset.crisis.notnull()]
dataset = dataset.loc[:,~dataset.columns.duplicated()]

#############create dataset with dynamic correlation

dyncorr=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dynamic_corr_dollar_eur.csv")
wei_dyn_corr =pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dynamic_corr_wei_dollar_eur.csv")
dollar_eur_gen_trend= pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\gen_dollar_eur_trend.csv")
dollar_eur_wei_trend= pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\wei_dollar_eur.csv")

n_start_obs = len(dataset) - len(wei_dyn_corr)

dataset2=dataset.iloc[n_start_obs:]

dataset2['wei_dyn_corr']=wei_dyn_corr.values[:,1]
dataset2['dynamic_corr']=dyncorr.values[:,1]
dataset2['dollar_eur_gen_trend']=dollar_eur_gen_trend.values[n_start_obs:,1]
dataset2['wei_dollar_eur_gen_trend']=dollar_eur_wei_trend.values[n_start_obs:,1]


dataset2['absdiiff'] = abs(dataset2.diff_wei_trend)/100

#dataset2 = dataset2.loc[:,~dataset2.columns.duplicated()]
#dataset2 = dataset2.drop('date.1',axis='columns')
dataset2 = dataset2.drop(dataset2.columns[1],axis='columns')

dataset2.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dataset2_dollar_eur.csv")

dataset2.columns=['Unnamed: 0', 'date', 'open_dollar_eur', 'high_dollar_eur', 'low_dollar_eur','dollar_eur', 'dollar_eur_adj', 'volume_dollar_eur', 'divid', 'split coef', 'dollar_eur', 'iphone', 'dollar_eur_watch', 'airpods', 'wei_dyn_corr',       'dynamic_corr', 'dollar_eur_gen_trend', 'wei_dollar_eur_trend']


dataset2['weighted_dollar_eur_trend'] = 0.65*dataset2.dollar_eur + 0.20*dataset2.iphone+ 0.075*dataset2.dollar_eur_watch +0.075*dataset2.airpods
dataset2['diff_wei_trend']=dataset2.wei_dollar_eur_gen_trend.diff()

wei_dollar_eur_trend=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\wei_dollar_eur.csv")
wei_dollar_eur_trend.columns=['n','wei_dollar_eur_trend']
wei_dollar_eur_trend = wei_dollar_eur_trend.head(len(dataset2)- len(wei_dollar_eur_trend))

dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dataset2_dollar_eur.csv")
dataset2['wei_dollar_eur_trend']=wei_dollar_eur_trend.wei_dollar_eur_trend











############################################## import result##############################################à

result=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\result.csv")
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
fig = plt.figure('portfolio --- dollar_eur')
plt.title('portfolio --- dollar_eur')
ax1 = fig.add_subplot(111)
ax1.plot( dataset2.date,result.port_value)
ax1.set_ylabel('port_value $')

ax2 = ax1.twinx()
ax2.plot(dataset2.dollar_eur, 'r-')
ax2.set_ylabel('dollar_eur $', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')



##############correlation

port_dollar_eur_corr= np.corrcoef(result.port_value, dataset2.dollar_eur)
