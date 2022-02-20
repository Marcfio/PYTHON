# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:03:40 2021

@author: MARCOFIORAVANTIPC
"""
#

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
import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt

# generate 20 data points
N = 20
X = np.random.random(N)*10-5
Y = 0.5*X-1+np.random.randn(N)
plt.scatter(X,Y);

###Linear Model
model = nn.Linear(1,1)
#Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# in ML we want our data to be of shape
# num_sample x num_dimensions

X = X.reshape(N,1)
Y = Y.reshape(N,1)


### input for pytorch --> float 32

inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

type(inputs)

#Train the model

n_epochs = 30
losses = []

for it in range(n_epochs):
    # zero the parameter gradients
    optimizer.zero_grad()
    
    #Forward pass
    outputs = model(inputs)
    loss = criterion(outputs,targets)
    
    #keep the loss so we can plot it later
    losses.append(loss.item())
    
    #Backward and optimize
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {it+1}/{n_epochs}, loss: {loss.item():.4f}')

plt.plot(losses);

predicted = model(inputs).detach().numpy()
plt.scatter(X,Y, label = 'Original data')
plt.plot(X,predicted, label = 'Fitted line')
plt.legend()
plt.show()


with torch.no_grad():
    out = model(inputs).numpy()
 
out


w = model.weight.data.numpy()
b = model.bias.data.numpy()
print(w,b)








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
