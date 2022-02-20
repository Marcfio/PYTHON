# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:43:33 2020

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




dataset2=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\dataset2_apple.csv")
result=pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\result.csv")
result.columns=['date','position','cash','ret']
result['port_value']=result.position+ result.cash

transactions = pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 4\dataset\transactions.csv")



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


neg_ret=0
pos_ret=0

for i in range(1, len(result)):
    if result.position[i-1]>=0:
        pos_ret=pos_ret+(result.ret[i]*result.port_value[i-1])
    elif result.position[i-1] < 0 :
        neg_ret=neg_ret+(result.ret[i]*( result.port_value[i-1]))
        
        