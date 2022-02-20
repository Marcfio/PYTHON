
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:20:12 2020

@author: MARCOFIORAVANTIPC
"""


from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
import threading
import time
import datetime
import pandas as pandas
import pickle


class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
    def historicalData(self, reqId, bar):
         print(f'Time: {bar.date} Close: {bar.close}')
         app.data.append([bar.date, bar.close])

def run_loop():
    app.run()

app = TestApp()
app.connect('127.0.0.1', 7497, 123)

#Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(1) #Sleep interval to allow time for connection to server

#Create contract object

Ticker= ['AMD','TSLA','AMZN','AAPL','ZNGA','NVDA','MSFT','JD','CSCO','FB']
#df=[AMD,TSLA,AMZN,AAPL,ZNGA,NVDA,MSFT,JD,CSCO,FB]
#df= ['AMD','TSLA','AMZN','AAPL','ZNGA','NVDA','MSFT','JD','CSCO','FB']
#df={'AMD':AMD,'TSLA':TSLA,'AMZN':AMZN,'AAPL':APPL,'ZNGA':ZNGA,'NVDA':NVDA,'MSFT':MSFT,'JD':JD,'CSCO':CSCO,'FB':FB}
df= list ()
appl_contract = Contract()
for i in range(len(Ticker)):
    appl_contract.symbol = Ticker[i]
    appl_contract.secType = 'STK'
    appl_contract.exchange = 'SMART'
    appl_contract.primaryExchange = 'NASDAQ'
    appl_contract.currency = 'USD'
    
    
    app.data = [] #Create empty variable to store candles
    #queryTime = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime("%Y%m%d %H:%M:%S")
    #Request historical candles
    app.reqHistoricalData(1, appl_contract, "", '120 D', '1 hour', 'MIDPOINT', 0, 2, False, [])
    
    time.sleep(10) #sleep to allow enough time for data to be returned
    
    #Working with Pandas DataFrames

    
    a = pandas.DataFrame(app.data, columns=['DateTime', 'Close'])
    a['DateTime'] = pandas.to_datetime(a['DateTime'], unit='s') 

    df.append(a)
    
   #df.to_csv('EURUSD_Hourly.csv')  
    #print(df)
   
time.sleep(1)


app.disconnect()

import pickle

with open('price_1.txt','wb') as fp:
    pickle.dump(df, fp)
    
#######################Dati di un indice###################à

Ticker=['SPX','NDX',]
Exchange=['CBOE','NASDAQ']
df= list ()
appl_contract = Contract()
for i in range(len(Ticker)):
    
    appl_contract.symbol = Ticker[i]

    appl_contract.secType = 'IND'
    #appl_contract.LastTradeDateOrContractMonth = 202003
    appl_contract.exchange = Exchange[i]
    #appl_contract.primaryExchange = 'NASDAQ'
    appl_contract.currency = 'USD'
    
    
    app.data = [] #Create empty variable to store candles
    app.reqHistoricalData(1, appl_contract, "", '120 D', '1 day', 'TRADES', 0, 2, False, [])    #requet daily data
    time.sleep(10) #sleep to allow enough time for data to be returned
    
    #Working with Pandas DataFrames

    
    a = pandas.DataFrame(app.data, columns=['DateTime', 'Close'])
    a['DateTime'] = pandas.to_datetime(a['DateTime'], unit='s') 

    df.append(a)
    
    #df.to_csv('EURUSD_Hourly.csv')  
    #print(df)
   
time.sleep(10)


app.disconnect()

import pickle

with open('price_index.txt','wb') as fp:
        pickle.dump(df, fp)
        
    
    