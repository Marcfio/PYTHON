# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:14:18 2021

@author: MARCOFIORAVANTIPC
"""



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# make the original data
N = 1000
series = np.sin(0.1*np.arange(N)) # + np.random.randn(N)*0.1

# plot it
plt.plot(series)
plt.show()


# build the dataset

T = 10
X = []
Y = []
for t in range (len(series) - T):
    x = series[t:t + T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1,T,1) # the last parameters "1" is the third dimension --> different from AR_MODEL
Y = np.array(Y).reshape(-1,1)
N = len(X)
print("X.shape",X.shape,"Y.shape", Y.shape)

#set device --> CUDA IS NOT AVAIABLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


### define simple RNN

class SimpleRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(SimpleRNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers
        
        # note: batch_first = true
        # applies the convention that our data will be of shape:
        # (num_samples, sequence_lenth, num_features)
        # rather than:
        # (sequence_length, num_samples, num_features)
        
        self.rnn = nn.RNN(
            input_size = self.D,
            hidden_size = self.M,
            num_layers = self.L,
            nonlinearity = 'relu',
            batch_first = True)
        self.fc = nn.Linear(self.M, self.K)
        
    def forward(self, X):
        # initial hidden states
        h0 = torch.zeros(self.L, X.size(0), self.M)#.to(device)
        
        # get RNN unit output
        # out is of size (N,T,M)
        # 2nd return value is hidden states at each hidden layer
        # we don't need thos now
        
        out, _ = self.rnn(X,h0)
        
        # we only want h(T) at the final time step
        # N x M -> N x K
        out = self.fc(out[:,-1,:])
        return out

# Instantiate the model
model = SimpleRNN(n_inputs=1, n_hidden=5, n_rnnlayers = 1, n_outputs=1)
#model.to(device)


#Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

#Make inputs and targets
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
X_test = torch.from_numpy(X[:-N//2].astype(np.float32))
y_test = torch.from_numpy(Y[:-N//2].astype(np.float32))


# #move data to gpu che non ho
# X_train, y_train = X_train.to(device), y_train.to(device)
# X_test, y_test = X_test.to(device), y_test.to(device)


# Training
def full_gd(model,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs = 200):
    
    # stuff to store
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    
    for it in range(epochs):
        #zero the parameters gradients
        optimizer.zero_grad()
        
        #Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        #Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Save losses
        train_losses[it] = loss.item()
        
        #Test loss
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[it] = test_loss.item()
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
                
        if (it+1)% 5 == 0:
            print(f'Epoch {it+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        
    return train_losses, test_losses

train_losses, test_losses = full_gd(model,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test)
    
## plot the train loss and test loss per iteration
plt.plot(train_losses, label = 'train loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()



# "Wrong" forecast using true targets

validation_target = Y[-N//2:]
validation_predictions = []


# index of first validation input
i = 0

while len(validation_predictions)< len(validation_target):
    input_ = X_test[i].reshape(1,T,1)
    p = model(input_)[0,0].item() # 1x1 array -> scalar
    i += 1
    
    # update the predictions list
    validation_predictions.append(p)
    
plt.plot(validation_target, label= 'forecast target')
plt.plot(validation_predictions, label = 'forecast prediction')
plt.legend()

# Forecast future values (use only self-predictions for making future predictions)

validation_target = Y[-N//2:]
validation_predictions = []
    

# last train input
# last_x = torch.from_numpy(X[-N//2]) # 1-D array of length T
# last_x = torch.from_numpy(X[-N//2]).astype(np.float32))
last_x = X_test[0].view(T)

while len(validation_predictions) < len(validation_target):
    input_ = last_x.reshape(1,T,1)
    p = model(input_)
    # [0,0] # 1x1 array -> scalar
    
    # update the predictions list
    validation_predictions.append(p[0,0].item())
    
    #make the new input
    last_x = torch.cat((last_x[1:], p[0]))

plt.plot(validation_target, label = 'forecast target')
plt.plot(validation_predictions, label = 'forecast prediction')
plt.legend()