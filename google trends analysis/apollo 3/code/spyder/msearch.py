
"""
Created on Sat Mar 14 21:07:48 2020

@author: MARCOFIORAVANTIPC
"""



import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from pytrends.request import TrendReq
from datetime import date
from datetime import timedelta
from datetime import datetime
from datetime import time
from pytrends import dailydata


def obtaindatah(kw_list,ndaybefore):
    end=date.today()
    start=end-timedelta(ndaybefore) 

                
    pytrends= TrendReq(hl='en-Us', tz=360 , retries=2, backoff_factor=0.1)
    df_h=pytrends.get_historical_interest(kw_list, year_start=start.year, month_start=start.month, day_start=start.day, year_end=end.year, month_end=end.month,day_end=end.day, cat=0, geo='', gprop='', sleep=3)
    return df_h


def obtaindatah2(kw_list,startday,endday):
    start=startday
    end=endday
                
                
    pytrends= TrendReq(hl='en-Us', tz=360 , retries=2, backoff_factor=0.1)
    df_h=pytrends.get_historical_interest(kw_list, year_start=start.year, month_start=start.month, day_start=start.day, year_end=end.year, month_end=end.month,day_end=end.day, cat=0, geo='', gprop='', sleep=3)
    return df_h


def dailydata1(kw_list,ndaybefore):
    end=date.today()
    start=end-timedelta(ndaybefore) 
    jimmi = pd.DataFrame()
    for i in range (len(kw_list)):
     data = dailydata.get_daily_data(kw_list[i], start.year, start.month, end.year, end.month)
     sam = np.zeros(len(data))
     sam[:]=data.values[:,4]
     jimmi[kw_list[i]]=sam
    
    jimmi['DateTime'] = data.index.strftime("%Y-%M%-%D %H:%M:%S")

    return jimmi


def dailydata2(kw_list,startday, endday):
    start=startday
    end=endday
    jimmi = pd.DataFrame()
    for i in range (len(kw_list)):
         data = dailydata.get_daily_data(kw_list[i], start.year, start.month, end.year, end.month)
         sam = np.zeros(len(data))
         sam[:]=data.values[:,4]
         jimmi[kw_list[i]]=sam
         jimmi.index = data.index

    return jimmi


def count_day (df):
    ja=list()
    ja.append(date(df.index[1].year,df.index[1].month,df.index[1].day))
    k=0
    i=0
    for i in range (0,len(df)-1):
        if df.index[i].day != df.index[i+1].day:
            ja.append(date(df.index[i+1].year,df.index[i+1].month,df.index[i+1].day))
     
    return ja       
            
  
def insertvalue(df,days,kw_list):
   
    r=len(kw_list)
    s=[len(days)]
    df_d=np.zeros(len(days))
    temp=pd.DatetimeIndex(days)
    df_final=pd.DataFrame(index=[temp])
    for m in range (0,r):
    
        values= df.values[:,m]
        varname=df.columns[m]
        b=df.index.date
        df_att=pd.DataFrame(values,columns = [varname],index=[b])
        lk=0  #lk non dovrebbe mai superare il numero di elementi di 'days'
        semisum=df_att.values[0]    #somma temporanea dei valori per effettuare la media giornaliera       
        jm_temp=1    # variabile temporanea per avere il numero di elementi in un giorno
        for i in range (0,len(df_att)-1):
                            
               if df_att.index[i] != df_att.index[i+1]:   #sommo valori per lo stesso giorno altrimenti passo a quello dopo
                   
                        df_d[lk]=semisum/jm_temp
                        semisum=df_att.values[i+1]
                        jm_temp=1
                        lk=lk+1
                        if lk == (len(days) - 1):
                            df_d[lk]=(sum(df_att.values[(i+1) : ]))/len([df_att[i+1:]])
                      
               else: 
                    semisum=semisum+ df_att.values[i+1]
                    if (df_att.values[i+1] !=0 ):   ####se il valore è zero non lo includo nella media
                        jm_temp=jm_temp+1
                    else:
                        jm_temp=jm_temp
                        
        df_final[kw_list[m]]=df_d
    
    df_final['DateTime']=temp
        
   
    return df_final   
  

    
  
def data (kw_list,selectdate,ndaybefore,startday,endday) :
    df_h=obtaindatah(kw_list,selectdate,ndaybefore,startday,endday)
    days= count_day(df_h)
    dfl=insertvalue(df_h,days,kw_list)
                    
    return dfl


