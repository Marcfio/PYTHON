# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 13:54:06 2020

@author: MARCOFIORAVANTIPC
"""


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
import scipy 
import mystat
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange as fe

##################################################################################################
###########################            BIT/USD            ########################################

bit_usd = pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\metatrader\ BTCUSD.csv")

ret_bit_usd = np.zeros(len(bit_usd))
for i in range (1,len(bit_usd)):
    ret_bit_usd[i]=bit_usd.close[i]-bit_usd.close[i-1]

plt.plot(ret_bit_usd)

exp_ret = np.average(ret_bit_usd[1:])
ret_bit_usd[0] = exp_ret

resid = ret_bit_usd - exp_ret
resid_2 = np.power(resid,2)
dataset_0 = pd.DataFrame()
dataset_0['returns'] = ret_bit_usd
dataset_0['resid'] = resid
dataset_0['resid_2'] = resid_2
dataset_0['volume'] = bit_usd.volume
corr_0 = dataset_0.corr()
#########lag correlation volume - returns
corr_vol_ret = mystat.lag_corr(bit_usd.volume, ret_bit_usd,20)
plt.title("right side: effect of volume -> returns")
plt.bar(corr_vol_ret.index,corr_vol_ret.lag_corr)


#########lag correlation volume - residuals
corr_vol_res2 = mystat.lag_corr(bit_usd.volume, resid_2 ,20)
plt.title("right side: effect of volume -> squared residuals")
plt.bar(corr_vol_res2.index,corr_vol_res2.lag_corr)


Vret = ret_bit_usd * bit_usd.volume/3000
plt.figure()
plt.plot(Vret)
plt.plot(ret_bit_usd)


Vprice = np.zeros(len(Vret))

for i in range (1,len(Vret)-1):
        Vprice[0]=bit_usd.close[0]
        Vprice[i]= Vprice[i-1]+Vret[i]
    
    
    
plt.figure()
plt.plot(Vprice)
plt.plot(bit_usd.close)

##################################################################################################
###########################           US30                ########################################
key = '5VGPLNQ2C2SYC0V7'
ts = TimeSeries(key, output_format= 'pandas')
fx = fe(key, output_format='pandas')

us30 , meta_data = ts.get_intraday(symbol='MSFT',interval='30min', outputsize='full')
(us30.columns)=['open','high','low','close','volume']


ret_us30 = np.zeros(len(us30))
for i in range (1,len(us30)):
    ret_us30[i]=us30.close[i]-us30.close[i-1]

plt.plot(ret_us30)

exp_ret = np.average(ret_us30[1:])
ret_us30[0] = exp_ret

resid = ret_us30 - exp_ret
resid_2 = np.power(resid,2)
dataset_0 = pd.DataFrame()
dataset_0['returns'] = ret_us30
dataset_0['resid'] = resid
dataset_0['resid_2'] = resid_2
dataset_0['volume'] = us30.volume
corr_0 = dataset_0.corr()
#########lag correlation volume - returns
corr_vol_ret = mystat.lag_corr(us30.volume, ret_us30,20)
plt.title("US30 -- right side: effect of volume -> returns")
plt.bar(corr_vol_ret.index,corr_vol_ret.lag_corr)


#########lag correlation volume - residuals
corr_vol_res2 = mystat.lag_corr(us30.volume, resid_2 ,20)
plt.title("US30 -- right side: effect of volume -> squared residuals")
plt.bar(corr_vol_res2.index,corr_vol_res2.lag_corr)


Vret = ret_us30 * us30.volume/3000
plt.figure()
plt.plot(Vret)
plt.plot(ret_us30)


Vprice = np.zeros(len(Vret))

for i in range (1,len(Vret)-1):
        Vprice[0]=us30.close[0]
        Vprice[i]= Vprice[i-1]+Vret[i]
    
    
    
plt.figure()
plt.plot(Vprice)
plt.plot(us30.close)
