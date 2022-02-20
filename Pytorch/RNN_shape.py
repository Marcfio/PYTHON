# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 23:38:54 2021

@author: MARCOFIORAVANTIPC
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Things you should  automatically know and have memorized
# N = # of samples
# T = sequence length
# M = # of hidden state
# D = # of input
# K = # of output units

# Make some data
N = 1
T = 10
D = 3
M = 5
K = 3
X = np.random.randn(N,T,D)

# Make an RNN
class SimpleRNN(nn.Module):
    def __init__(self,n_input, n_hidden, n_output):
        super(SimpleRNN, self).__init__()
        self.D = n_input
        self.K = n_output
        self.M = n_hidden
        self.rnn = nn.RNN(
            input_size = self.D, 
            hidden_size = self.M,
            nonlinearity = 'tanh',
            batch_first= True)
        self.fc = nn.Linear(self.M, self.K)
        
        
    def forward(self,X):
        # initial hidden state
        h0 = torch.zeros(1,X.size(0), self.M)
        
        # get RNN unit output
        out, _ = self.rnn(X, h0)
        
        # we only want h(T) at the final step
        # out = self.fc(out[:, -1, :])
        out = self.fc(out)
        return out

# Instantiate the model
model = SimpleRNN(n_input= D,n_hidden = M, n_output = K )

#Get the output
inputs = torch.from_numpy(X.astype(np.float32))
out = model(inputs)        
out
        
out.shape        
        
#save for later
Yhats_torch = out.detach().numpy()

W_xh, W_hh, b_xh, b_hh = model.rnn.parameters()

W_xh.shape 
W_xh
W_xh = W_xh.data.numpy()
W_xh
b_xh = b_xh.data.numpy()
W_hh = W_hh.data.numpy()
b_hh = b_hh.data.numpy()


# Did we do right
W_xh.shape , W_hh.shape, b_hh.shape, b_xh.shape

# Now get the FC layer weights
Wo, bo = model.fc.parameters()

Wo = Wo.data.numpy()
bo = bo.data.numpy()
Wo.shape, bo.shape, b_hh


# see if we can replicate the output ---> y_t = sigma[W_o * h_t + b_0]... h_t = sigma[W_xh T x_t + W_hh T h_t-1 + b_h]
h_last = np.zeros(M)
x = X[0]
Yhats = np.zeros((T,K))

for t in range(T): 
    h = np.tanh(x[t].dot(W_xh.T)+ b_xh + h_last.dot(W_hh.T) + b_hh)
    y = h.dot(Wo.T) + bo
    Yhats[t] = y
    
    h_last = h
    
print (Yhats)
    
#Check

np.allclose(Yhats, Yhats_torch )
    
    







            
            
            
            