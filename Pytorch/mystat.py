# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:45:45 2020

@author: MARCOFIORAVANTIPC
"""
import numpy as np
import pandas as pd

def lag_corr (x1,x2,lag):
    lag_corr= np.zeros(lag*2+1)
    index = np.zeros(lag*2 +1)
    
    if (len(x1) != len(x2)) :
        print("dimensioni differenti")
    else:
         for i in range(0,lag+1) : #sinistra del grafico
             corr_temp =np.corrcoef(x1[0:len(x1)-i],x2[i:])
             corr_sing = corr_temp[0,1]
             ##### a destra stai anticipando x1 a sinistra x2----> a destra vedi gli effetti di x1 su x2 e viceversa
             lag_corr[lag+i] = corr_sing
             index[lag+i] = i
         for j in range(1,lag+1):# destra del grafico
             corr_temp = np.corrcoef( x2[0:len(x1)-j] , x1[j:] )
             corr_sing = corr_temp[0,1]
             ##### a destra stai anticipando x1 a sinistra x2----> a destra vedi gli effetti di x1 su x2 e viceversa
             lag_corr[lag - j] = corr_sing
             index[lag-j] = -j
             
    correlation_lag = pd.DataFrame()
    correlation_lag['lag_corr'] = lag_corr
    correlation_lag.index = index
    return correlation_lag