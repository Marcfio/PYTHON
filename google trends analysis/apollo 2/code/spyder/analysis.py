# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:40:54 2020

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


############import from csv file ###############################################################################


oil_trends=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\oil_trends_daily.csv")
data_oil=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\oil.csv")
dataset=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dataset_oil.csv")

dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dataset2_oil.csv")
result=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil.csv")

#############result analysis ###########################################################################àà

result.columns=['date','position','cash','ret']
result['port_value']=result.position+result.cash


dataset.plot(x = dataset2.columns[1] , y= ['barrel','oil','wti'],title='google trends data', fontsize=8)

neg_ret=0
pos_ret=0

for i in range(1, len(result)):
    if result.position[i-1]>=0:
        pos_ret=pos_ret+(result.ret[i]*result.port_value[i-1])
    elif result.position[i-1] < 0 :
        neg_ret=neg_ret+(result.ret[i]*( result.port_value[i-1]))
        










plt.figure()
plt.title('return of the strategy')
plt.plot(dataset2.date.values , result.ret)

#result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil.csv")
#transactions.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\transaction_oil.csv")


plt.figure()
plt.plot(result.date , result.port_value)
plt.plot( dataset2.oil_price)
plt.show()
plt.close()



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


