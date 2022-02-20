# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 22:09:53 2021

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
import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt


bit_usd = pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\metatrader\ BTCUSD.csv")

ret_bit_usd = np.zeros(len(bit_usd))
for i in range (1,len(bit_usd)):
    ret_bit_usd[i]=bit_usd.close[i]-bit_usd.close[i-1]
    
    
    
Y_1 = ret_bit_usd.reshape(-1,1)
X = bit_usd.values[:,1].reshape(-1,1)
Y = bit_usd.values[:,3].reshape(-1,1)


plt.scatter(X,Y)

mx = X.mean()
sx = X.std()
my = Y.mean()
sy = Y.std()
X = (X - mx)/ sx
Y = (Y - my)/ sy

plt.scatter(X,Y)

X = X.astype(np.float32)
Y = Y.astype(np.float32)


model = nn.Linear(1,1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.7)

inputs = torch.from_numpy(X)

targets = torch.from_numpy(Y)


n_epochs = 100
losses = []

for it in range(n_epochs):
    
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    w = []
    
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {it+1}/{n_epochs}, Loss: {loss.item():.4f}')
    w_temp = model.weight.data.numpy()
    w_temp = w_temp.astype(np.float32)
    w.append(w_temp.item())    
    
plt.plot(losses)


predicted = model(torch.from_numpy(X)).detach().numpy()
plt.plot(X,Y, 'ro', label = 'Original data')
plt.plot(X, predicted, label = 'Fitted line')
plt.legend()
plt.show()


plt.plot(w)
