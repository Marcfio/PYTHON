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
import statsmodels.api as sm
from statsmodels.tsa.api import VAR ,  SVAR
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Y5VGPLNQ2C2SYC0V7
key = '5VGPLNQ2C2SYC0V7'
ts = TimeSeries(key, output_format= 'pandas')



##################Market data##########################
data_oil_day, meta_data_day = ts.get_daily_adjusted(symbol='OIL',outputsize='full')
(data_oil_day.columns)=['open_oil','high_oil','low_oil','oil_price','oil_adj','volume_oil','divid','split coef']
data_oil_day.to_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\oil.csv')



data_oil = data_oil_day.head(900)

plt.plot(data_oil.oil)


#####################Google trends data####################
kw_list_oil=["oil","production","wti","brent","crude oil","barrel"]

startday = datetime(2017,1,1)
endday = date.today()
oil_trends= msearch.dailydata2(kw_list_oil,startday,endday)
oil_trends.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\oil_trends_daily.csv")




##### merge dataset ########

oil_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\oil_trends_daily.csv")
oil_trends=oil_trends.sort_values(by="date")
oil_trends.index= oil_trends.date
#oil_trends=oil_trends.drop(columns='date')


data_oil=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\oil.csv")
data_oil=data_oil.sort_values(by="date", ascending = True)
data_oil.index=data_oil.date
#data_oil=data_oil.drop(columns='date')


dataset = pd.concat([data_oil,oil_trends],axis=1, join='outer')
dataset=dataset[dataset.oil_price.notnull()]
dataset=dataset[dataset.oil.notnull()]
dataset = dataset.loc[:,~dataset.columns.duplicated()]

dataset.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dataset_oil.csv")
dataset.index=dataset.date

############import from csv file ###############################################################################


oil_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\oil_trends_daily.csv")
data_oil=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\oil.csv")
dataset=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dataset_oil.csv")




dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dataset2_oil.csv")
#dataset2.columns[1]='date'
#dataset2=dataset2.drop(columns= dataset2.columns[0]) ##run 2 times





#dataset=dataset.drop(dataset.columns[[0]],axis='columns')



#############create dataset with dynamic correlation########################################

dyncorr=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dynamic_corr2.csv")
dyncorr_2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dynamic_corr2_gen.csv")

n_start_obs = len(dataset) - len(dyncorr)

dataset2=dataset.iloc[n_start_obs:]
dataset2['gen_trend'] = 0.65 * dataset2.oil +  0.25 * dataset2.barrel + 0.05 * dataset2.wti + 0.05 * dataset2.brent
dataset2['dynamic_corr']=dyncorr.values[:,1]
dataset2['dynamic_corr_gen']=dyncorr_2.values[:,1]


dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dataset2_oil.csv")
dataset2['diff_gen_trend']=dataset2.gen_trend.diff()
#dataset2 = dataset2.loc[:,~dataset2.columns.duplicated()]
#dataset2 = dataset2.drop(dataset2.columns[0],axis='columns')
#dataset2.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dataset2_oil.csv")




# #########columns manip##########
# cols = dataset2.columns.tolist()
# cols=cols[-1:] + cols[:-1]
# dataset2=dataset2[cols]

# dataset2=dataset2.drop(columns= dataset2.columns[1])




############################################## import result######################

result=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil.csv")
result.columns=['date','position','cash','ret']
result['port_value']=result.position+result.cash
#result.to_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oilstrategy.csv')


#result_barr=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil_barr.csv")
#result_barr.columns=['date','position','cash','ret']

#################################some plots#########################

dataset.plot(x = dataset2.columns[1] , y= ['barrel','oil','wti'],title='google trends data', fontsize=8)


plt.figure()
plt.title('return of the strategy')
plt.plot(dataset2.date.values , result.ret)

result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil.csv")
transactions.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\transaction_oil.csv")


plt.figure()
plt.plot(result.date , result.port_value)
plt.plot( dataset2.oil_price)
plt.show()
plt.close()

result_barr.plot(x=result_barr.columns[0] , y = ['return'])


##############plot varible with different scale on y axis###à
fig = plt.figure('portfolio --- oil_price')
plt.title('portfolio --- oil_price')
ax1 = fig.add_subplot(111)
ax1.plot( dataset2.date,result.port_value)
ax1.set_ylabel('port_value $')

ax2 = ax1.twinx()
ax2.plot( dataset2.oil_price, 'r-')
ax2.set_ylabel('oil_price $', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')


