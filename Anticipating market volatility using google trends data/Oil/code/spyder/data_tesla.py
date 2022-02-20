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
data_tesla_day, meta_data_day = ts.get_daily_adjusted(symbol='TSLA',outputsize='full')
(data_tesla_day.columns)=['open_tsla','high_tsla','low_tsla','tsla_price','tsla_adj','volume_tsla','divid','split coef']
data_tesla_price = data_tesla_day.head(900)

data_tesla_day.to_csv(r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\tesla_data.csv')




plt.plot(data_tesla_price.tsla_price)


#####################Google trends data####################
kw_list_tesla=["tesla","musk","elon",]

startday = datetime(2017,1,1)
endday = date.today()
tesla_trends= msearch.dailydata2(kw_list_tesla,startday,endday)
tesla_trends.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\tesla_trends_daily.csv")

plt.plot(tesla_trends.index, tesla_trends.elon)
plt.plot
##### merge dataset ########

tesla_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\tesla_trends_daily.csv")
tesla_trends=tesla_trends.sort_values(by="date")
tesla_trends.index= tesla_trends.date


data_tesla=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\tesla_data.csv")
data_tesla=data_tesla.sort_values(by="date", ascending = True)
data_tesla.index=data_tesla.date


dataset = pd.concat([data_tesla,tesla_trends],axis=1, join='outer')
dataset=dataset[dataset.tsla_price.notnull()]
dataset=dataset[dataset.tesla.notnull()]
dataset = dataset.loc[:,~dataset.columns.duplicated()]
dataset.index=dataset.date

dataset.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\dataset_tesla.csv")


############import from csv file ###############################################################################


tesla_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\tesla_trends_daily.csv")
data_tesla=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\tesla_data.csv")
dataset=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\dataset_tesla.csv")




dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\dataset2_tesla.csv")


#dataset=dataset.drop(dataset.columns[[0]],axis='columns')



#############create dataset with dynamic correlation

dyncorr=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\dynamic_corr2.csv")
n_start_obs = len(dataset) - len(dyncorr)

dataset2=dataset.iloc[n_start_obs:]
dataset2['dynamic_corr']=dyncorr.values[:,1]
dataset2['tesla_gen_trend']=dataset2.tesla+dataset2.elon

dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\dataset2_tesla.csv")

#dataset2 = dataset2.loc[:,~dataset2.columns.duplicated()]
#dataset2 = dataset2.drop('date.1',axis='columns')
#dataset2 = dataset2.drop(dataset2.columns[0],axis='columns')

dataset2.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\dataset2_tesla.csv")

dataset.plot(x = 'date' , y = [ ])
############################################## import result

result=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\result.csv")
result.columns=['date','position','cash','ret']
result['port_value']=result.position+ result.cash

####export dataset##############################################################
#result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\result_oil.csv")
#transactions.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 3\dataset\transaction_oil.csv")

#################################some plots#########################
dataset2.plot(x = dataset2.columns[1] , y= ['tesla','elon','tesla_gen_trend'],title='google trends data', fontsize=8)


plt.figure()
plt.title('return of the strategy')
plt.plot(dataset2.date.values , result.ret)



plt.figure()
plt.title('portfolio value')
plt.plot(result.date , result.port_value)
#plt.plot( dataset2.tsla_price)




##############plot varible with different scale on y axis###à
fig = plt.figure('portfolio --- tesla_price')
plt.title('portfolio --- tesla_price')
ax1 = fig.add_subplot(111)
ax1.plot( dataset2.date,result.port_value)
ax1.set_ylabel('port_value $')

ax2 = ax1.twinx()
ax2.plot(dataset2.tsla_price, 'r-')
ax2.set_ylabel('tesla_price $', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')



##############correlation

port_tesla_corr= np.corrcoef(result.port_value, dataset2.tsla_price)
