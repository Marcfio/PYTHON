# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:34:58 2021

@author: MARCOFIORAVANTIPC
"""

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#make the roiginal data
series = np.sin((0.1*np.arange(400))**2)
# #plot it
# plt.plot(series)
# plt.show()

### build the dataset
# let's see if we can use T past values to predict the next value
T = 10
D = 1
X = []
Y = []

for t in range (len(series) - T):
    x = series[t:t +T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1,T) #☺ make it NxT
Y = np.array(Y).reshape(-1,1)
N = len(X)
print("X.shape", X.shape , "Y.shape", Y.shape)

### try autoregresive linear model
model = nn.Linear(T,1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)


# Make inputs and targets
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))

X_test = torch.from_numpy(X[-N//2:].astype(np.float32))
y_test = torch.from_numpy(Y[-N//2:].astype(np.float32))

# Training
def full_gd(model,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs = 200):
    
# Stuff to store
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    
    for it in range(epochs):
            #zero the parameter gradients
            optimizer.zero_grad()
            
            #forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Save losses
            train_losses[it] = loss.item()
            
            # Test loss
            test_outputs = model (X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses[it] = test_loss.item()
            
            if (it + 1) % 5 == 0 :
                print(f'Epch {it+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
                
    return train_losses, test_losses



train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train,X_test, y_test)

### Plot the train loss and test loss per iteration

plt.plot(train_losses, label = 'train loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()

## one-step forecast using true targets
## Note: even the one-step forecast fails badly
validation_target = Y[-N//2:]
validation_predictions = []

# index of first validation input
i = 0

while len(validation_predictions) < len(validation_target):
    input_ = X_test[i].reshape(1, -1)
    p = model(input_)[0,0].item() # 1x1 array -> scalar
    i += 1
    
    # update the predictions list
    validation_predictions.append(p)
    
    
plt.plot(validation_target, label = 'forecast target')
plt.plot(validation_predictions, label = 'forecast prediction')
plt.legend()


# Multi-step forecast
validation_target = Y[-N//2:]
# validation_predictions = []

# # last train input
# # last_x = torch.from_numpy(X[-N//2]) # 1-D array of length T

# last_x = torch.from_numpy(X[-N//2].astype(np.float32))


# Multi-step forecast
validation_target = Y[-N//2:]
validation_predictions = []

# last train input
# last_x = torch.from_numpy(X[-N//2]) # 1-D array of length T
last_x = torch.from_numpy(X[-N//2].astype(np.float32))

while len(validation_predictions) < len(validation_target):
    input_ = last_x.reshape(1,-1)
    p = model(input_)
    # [0,0] # 1x1 array -> scalar
    
    # update the predictions list
    validation_predictions.append(p[0,0].item())
    
    # make the new input
    last_x = torch.cat((last_x[1:], p[0]))
    
    
### Define RNN

class RNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(RNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers
        
        # note: batch_first = True
        # applies the convention that our data will be of shape:
        # (num_samples, sequence_length, num_features)
        # rather than:
        # (sequence_length, num_samples, num_features)
        self.rnn = nn.GRU(
            input_size = self.D,
            hidden_size = self.M,
            num_layers = self.L,
            batch_first = True)
        self.fc = nn.Linear(self.M, self.K)
        
    def forward(self,X):
        # initial hidden states
        h0 = torch.zeros(self.L, X.size(0), self.M)
        
        # get RNN unit output
        # out is of size (N,T,M)
        # 2nd return value is hidden states at each hidden layer
        # we don't need those now
        
        out, _ = self.rnn(X,h0)
        
        
        # we only want h(T) at the final time step
        out = self.fc(out[:,-1, :])
        return out
# Intantiate the model
model = RNN(n_inputs = 1, n_hidden = 10, n_rnnlayers = 1, n_outputs = 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

# Make inputs and targets
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))

X_test = torch.from_numpy(X[-N//2:].astype(np.float32))
y_test = torch.from_numpy(Y[-N//2:].astype(np.float32))

# Training
def full_gd(model,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs = 200):
    
# Stuff to store
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    
    for it in range(epochs):
            #zero the parameter gradients
            optimizer.zero_grad()
            
            #forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Save losses
            train_losses[it] = loss.item()
            
            # Test loss
            test_outputs = model (X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses[it] = test_loss.item()
            
            if (it + 1) % 5 == 0 :
                print(f'Epch {it+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
                
    return train_losses, test_losses



train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train,X_test, y_test)

            
            
            
            
               



















    
