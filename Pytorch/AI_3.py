# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:01:00 2021

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
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange as fe
import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer 


data = load_breast_cancer()

X,Y = data.data, data.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, = train_test_split(data.data, data.target, test_size = 0.33)
N,D = X_train.shape


############# standardize data  ###############################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)


model = nn.Sequential(nn.Linear(D,1), nn.Sigmoid())


##Loss and optimizer

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())


# torch tensor creation

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1,1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1,1))


#Train the model
n_epochs = 1000
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

for it in range(n_epochs):
    
    optimizer.zero_grad()
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    #Backward and optimize
    loss.backward()
    optimizer.step()
    
    #Get test loss    
    outputs_test = model(x_test)
    loss_test = criterion(outputs_test, y_test)
    
    # Save losses
    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()
    
    if (it + 1) % 50 == 0:
        print(f'Epoch {it+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {loss.test():')
#### Evaluate the model ---> we cannot use MSE (qualitative character)
##Get accuracy

#Plot the train loss and test loss per iteration
    plt.plot(train_losses, label = 'train loss')
    plt.plot(test_losses, label = 'test loss')
    plt.legend()
    plt.show()
with torch.no_grad():
    p_train = model(X_train)
    p_train = np.round(p_train.numpy)
    train_acc = np.mean(y_train.numpy() == p_train)
    
    p_test = model(X_test)
    p_test = np.round(p_test.numpy())
    test_acc = np.mean(y_test.nympy() == p_test)
    print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
    