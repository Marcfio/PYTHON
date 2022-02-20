# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:20:21 2020

@author: MARCOFIORAVANTIPC
"""

import pandas as pd
import backtrader as bt
import backtrader.feeds as btfeeds
from datetime import datetime
from datetime import date
import backtrader.plot
import matplotlib 
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo

################class for importing data########################################àààà
class GenericCSV_PE(btfeeds.GenericCSVData):

    # Add a 'pe' line to the inherited ones from the base class
    lines = ('dyn_corr','oil_trend','barrel_trend','gen_trend')
    
    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('dyn_corr', 8),('oil_trend',9),('barrel_trend',10),('gen_trend' , 11))


################################################### strategy n. 1 ###################################################
class sellbuy(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.oil_trend = data.oil_trend
        self.volume = data.volume
    def next(self):
        if not self.position:
            
                if self.dyn_corr[-1] < -0.25:
                    self.sell(size=100)
                elif self.dyn_corr[-1]>0.25:
                    self.buy(size=100)
 
        else:
            
                if self.position.size >0:

                     if self.dyn_corr[-1] < -0.10 and self.dyn_corr[-1] > -0.25:
                        self.close(size=100)
                     elif self.dyn_corr[-1] <= -0.25:
                         self.close(size=100)
                         self.sell(size=100)

                elif self.position.size<0:
                      if self.dyn_corr[-1]> 0.10 and self.dyn_corr[-1]< 0.25 :
                        self.close(size=100)
                  
                      elif self.dyn_corr[-1]>=0.25:
                          self.close(size=100)                
                          self.buy(size=100)
                    
                        
                        
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)     
                

################################################### strategy n. 2 ###################################################
                
class sell(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
    
    def next(self):
        if not self.position:
            if self.dyn_corr[-1] < -0.25:
                self.sell(size=100)
           

                 
                
        else:
              if self.dyn_corr[-1]>0  :
                self.close(size=100)
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)

################################################### strategy n. 3 ###################################################
        
class barr(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.oil_trend = data.oil_trend
        self.volume = data.volume
        self.barrel_trend = data.barrel_trend
    def next(self):
        if self.barrel_trend>42:    
            if not self.position:
             #   if self.trump_trend[-1]>0:
                    if self.dyn_corr[-1] < -0.1:
                        self.sell(size=100)
                    if self.dyn_corr[-1]>0.1:
                        self.buy(size=100)
     
            else:
               # if self.trump_trend[-1]>0:
                    if self.position.size >0:
    
                         if self.dyn_corr[-1] < -0.05:
                            self.close(size=100)
    
    
                    elif self.position.size<0:
                          if self.dyn_corr[-1]>0.05  :
                            self.close(size=100)
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)     
        
################################################### strategy n. 4 ###################################################
        
class everytimeisagoodtime(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.oil_trend = data.oil_trend
        self.volume = data.volume
    def next(self):
        if not self.position:
            
                if self.dyn_corr[-1] < -0.1:
                    self.sell(size=100)
                if self.dyn_corr[-1]>0.1:
                    self.buy(size=100)
 
        else:
            
                if self.position.size >0:

                     if self.dyn_corr[-1] < -0.08:
                        self.close(size=100)
                        self.sell(size=100)

                elif self.position.size<0:
                      if self.dyn_corr[-1]> 0.05  :
                        self.close(size=100)
                        self.buy(size=100)
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)     



################################################### strategy n. 4 ###################################################
        
class trend_base(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.oil_trend = data.oil_trend
        self.volume = data.volume
        self.gen_trend = data.gen_trend
    def next(self):
         if self.gen_trend>0:  
            if not self.position:
                
                    if self.dyn_corr[-1] < -0.1:
                        self.sell(size=5000)
                    if self.dyn_corr[-1]>0.1:
                        self.buy(size=5000)
     
            else:
                
                    if self.position.size >0:
    
                         if self.dyn_corr[-1] < -0.08:
                            self.close(size=5000)
                            self.sell(size=5000)
    
                    elif self.position.size<0:
                          if self.dyn_corr[-1]> 0.05  :
                            self.close(size=5000)
                            self.buy(size=5000)
            portvalue = cerebro.broker.getvalue()
        
            print('Portfolio Value: ${}'.format(portvalue))
            print(self.position.size)                  
        
startcash = 150000
cerebro = bt.Cerebro()
cerebro.addstrategy(trend_base)############################################ADD STRATEGY

#Get S&P500 from CSV
data = GenericCSV_PE(
    dataname= r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\dataset2_oil.csv',
    fromdate=datetime(2017, 1, 1),
    #todate=datetime(2019, 12, 31),
    nullvalue=0.0,
    dtformat=('%Y-%m-%d'),
    datetime = 1,
    open = 2,
    high = 3,
    low = 4,
    close = 5,
    adjusted = 6,
#     volume = 7,
    dyn_corr = 18,
    barrel_trend=15,
    oil_trend = 10,
    gen_trend = 19
        )

cerebro.adddata(data)

cerebro.broker.setcash(startcash)

cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
results = cerebro.run()




##########analyzing

strat = results[0]

pyfoliozer = strat.analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()


#cerebro.run()

#Get final portfolio Value
portvalue = cerebro.broker.getvalue()
pnl = portvalue - startcash

#Print out the final result
print('Final Portfolio Value: ${}'.format(portvalue))
print('Initial Portfolio Value: ${}'.format(startcash))
print('P/L: ${}'.format(pnl))

#Finally plot the end results
cerebro.plot(iplot=False)
cerebro.plot()

result = pd.concat([positions,returns],axis=1, join = 'outer')
result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil.csv")



import pyfolio as pf
pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions)
   # round_trips=True)
   
   
result = pd.concat([positions,returns],axis=1, join = 'outer')
result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil_egd.csv")










#result_barr = pd.concat([positions,returns],axis=1, join = 'outer')
#result_barr.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil_barr.csv")

#result_barr = pd.concat([positions,returns],axis=1, join = 'outer')
#result_barr.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 2\dataset\result_oil_barr.csv")

