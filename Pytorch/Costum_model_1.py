# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 22:09:13 2021

@author: MARCOFIORAVANTIPC
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# %%
bit_usd = pd.read_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\metatrader\ BTCUSD.csv")

ret_bit_usd = np.zeros(len(bit_usd))
for i in range (1,len(bit_usd)):
    ret_bit_usd[i]=bit_usd.close[i]-bit_usd.close[i-1]
    
    
Y_1 = ret_bit_usd.reshape(-1,1)
X = bit_usd.values[:,1].reshape(-1,1)
Y = bit_usd.values[:,3].reshape(-1,1)


plt.scatter(X,Y)

# mx = X.mean()
# sx = X.std()
# my = Y.mean()
# sy = Y.std()
# X = (X - mx)/ sx
# Y = (Y - my)/ sy

# plt.scatter(X,Y)

# x= X.astype(np.float32)
# Y = Y.astype(np.float32)
# %%
class ANN(nn.Module):
        def __init__(self):
            super(ANN,self).__init__()
            self.layer1 = nn.Linear(1,2999)
            self.layer2 = nn.ReLU()
            self.layer3= nn.Linear(2999,1)
        def forward(self,x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x

model = ANN()
criterion = nn.MSELoss()  #loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


# class Convolution(nn.Module):
#     def __init__(self,K):
#         super(CNN,self).__init()__
#         #define conv layer
#         self.layer1 = nn.Linear(1,2999)
        
        
    
#         #define sequential layer
#         self.conv = nn.Sequential(
#             nn.Conv2d(3,32, kernel_size = 3, stride = 2)
#             nn.Conv2d(32,64, kernel_size = 3, stride = 2)
#             nn.Conv2d(64,128, kernel_size = 3, stride = 2)
#             )
        
        
        
        

def full_gd(model,criterion,optimizer, X_train, y_train, epochs = 1000):
    train_losses= np.zeros(epochs)
    
    for it in range(epochs):
        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        #Backward and optimize
        loss.backward()
        optimizer.step()
        
        #Save losses
        train_losses[it] = loss.item()
        
        if (it+1)%50 == 0:
            print(f'Epoch {it+1}/{epochs}, Train Loss: {loss.item():.4f}')
            
    return train_losses


X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(Y.astype(np.float32))
train_losses = full_gd(model,criterion,optimizer,X_train,y_train.reshape(-1,1))
        
plt.plot(train_losses)         
output = model(X_train)
