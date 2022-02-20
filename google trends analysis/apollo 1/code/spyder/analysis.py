# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:31:34 2020

@author: MARCOFIORAVANTIPC
"""

import msearch
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR ,  SVAR
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def workdata(price_h,trend_h):
    
    
    trend_h['DateTime'] = trend_h.index.strftime("%Y-%M%-%D %H:%M:%S")
    ntime=0
    datetimeind = list()
    for i in range(len(trend_h)):
        for j in range (len(price_h)):
           if str (trend_h.DateTime[i]) == str (price_h.DateTime[j]):
               #ntime=ntime+1
               datetimeind.append(trend_h.DateTime[i])
    
    
    db=pd.DataFrame()
    for i in range (len(trend_h.columns)-2):
        db[trend_h.columns[i]]=()
    db[price_h.columns[1]]=()        
        
    
    
    value_temp = np.zeros([len(datetimeind),len(db.columns)])
    jim=0
    for i in range(len(trend_h)):
        for j in range (len(price_h)):
           if str (trend_h.DateTime[i]) == str (price_h.DateTime[j]):
             value_temp[jim,:-1]=trend_h.values[i,:-2]
             value_temp[jim,-1]=price_h.Close[j]
             jim=jim+1
             
    for i in range (len(db.columns)):
        db[db.columns[i]]= value_temp[: , i]
    
    db.Close = value_temp[:,-1]
    db['DateTime']=datetimeind
    
    return db

def workdata2(price_h,trend_h):
    price_h['DateTime'] = pd.DatetimeIndex(price_h.index)
    price_h.sort_values(by=['DateTime'], inplace = True  )

    trend_h['DateTime'] =  pd.DatetimeIndex(trend_h.index)
    trend_h.sort_values(by=['DateTime'], inplace = True  )

    ntime=0
    datetimeind = list()
    for i in range(len(trend_h)):
        for j in range (len(price_h)):
           if ( (trend_h.DateTime[i].hour) == (price_h.DateTime[j].hour)) &  ( (trend_h.DateTime[i].day) == (price_h.DateTime[j].day)) & ( (trend_h.DateTime[i].month) == (price_h.DateTime[j].month)):
               #ntime=ntime+1
               datetimeind.append(trend_h.DateTime[i])
 
    
    db=pd.DataFrame()
    for i in range (len(trend_h.columns)-2):
        db[trend_h.columns[i]]=()
    for j in range (len(price_h.columns)):
        db[price_h.columns[j]]=()        
   
    sam=len(price_h.columns)-1
    value_temp = np.zeros([len(datetimeind),len(db.columns)])
    jim=0
    for i in range(len(trend_h)):
       for j in range (len(price_h)):
            if ( (trend_h.DateTime[i].hour) == (price_h.DateTime[j].hour)) &  ( (trend_h.DateTime[i].day) == (price_h.DateTime[j].day)) & ( (trend_h.DateTime[i].month) == (price_h.DateTime[j].month)):
             value_temp[jim,: -(sam+1)]=trend_h.values[i,:-2]
             value_temp[jim, -sam : ]=price_h.values[j,: - 1]
             jim=jim+1
             
    for i in range (len(db.columns)):
        db[db.columns[i]]= value_temp[: , i]
   

    db['DateTime']=datetimeind
    
    return db

def workdata3(price_h,trend_h):
    price_h['DateTime'] = pd.DatetimeIndex(price_h.index)
    trend_h['DateTime'] =  pd.DatetimeIndex(trend_h.index)

    ntime=0
    datetimeind = list()
    for i in range(len(trend_h)):
        for j in range (len(price_h)):
            if ( ( (trend_h.DateTime[i].day) == (price_h.DateTime[j].day)) & ( (trend_h.DateTime[i].month) == (price_h.DateTime[j].month)) & ((trend_h.DateTime[i].year )==( price_h.DateTime[j].year))):
                         #ntime=ntime+1
               datetimeind.append(trend_h.DateTime[i])
 
    
    db=pd.DataFrame()
    for i in range (len(trend_h.columns)):
        db[trend_h.columns[i]]=()
    for j in range (len(price_h.columns)):
        db[price_h.columns[j]]=()        
    db=db.drop(columns='DateTime')
    sam=len(price_h.columns)-2
    value_temp = np.zeros([len(datetimeind),len(db.columns)])
    jim=0
    for i in range(len(trend_h)):
       for j in range (len(price_h)):
            if ( ( (trend_h.DateTime[i].day) == (price_h.DateTime[j].day)) & ( (trend_h.DateTime[i].month) == (price_h.DateTime[j].month)) & ((trend_h.DateTime[i].year )==( price_h.DateTime[j].year))):
             value_temp[jim,: -(sam+1)]=trend_h.values[i,:-1]
             value_temp[jim, -(sam+1) : ]=price_h.values[j,: - 1]
             jim=jim+1
             
    for i in range (len(db.columns)):
        db[db.columns[i]]= value_temp[: , i]
   

    db['DateTime']=datetimeind
     
    return db

