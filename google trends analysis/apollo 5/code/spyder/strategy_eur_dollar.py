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
    lines = ('dyn_corr','dollar_eur_trend', 'diff_wei_trend','abs_var')
    
    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (  ('dyn_corr', 8)   ,   ('dollar_eur_trend' , 9)   ,  ('diff_wei_trend' , 10) ,('abs_var' , 10) )


################################################### strategy n. 1 ###################################################
class firstStrategy(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
    
    def next(self):
        if not self.position:
            if self.dyn_corr[-1] > 0:
                self.buy(size=1)
                print(self.position.size)
        else:
            if self.dyn_corr[-1] < -0.3:
                self.sell(size=1)
                print(self.position.size)

################################################### strategy n. 2 ###################################################
                
class sellbuy(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
    
    def next(self):
        if not self.position:
            if self.dyn_corr[-1] < -0.50:
                self.sell(size=50)
            if self.dyn_corr[-1]>0.3:
                self.buy(size=50)
 

        else:
            if self.position.size >0:
                
                 if self.dyn_corr[-1] < -0.2:
                    self.close(size=50)
                    self.sell(size=50)
                    
                
            elif self.position.size<0:
                  if self.dyn_corr[-1]>0  :
                    self.close(size=50)
                    self.buy(size=50)
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)

################################################### strategy n. 3 ###################################################
                
class sellbuy2(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
    
    def next(self):
        if not self.position:
            if self.dyn_corr[-1] < -0.55:
                self.sell(size=1000)
            if self.dyn_corr[-1]>0.40:
                self.buy(size=1000)
 

        else:
            if self.position.size >0:
                
                 if self.dyn_corr[-1] < -0.1 and self.dyn_corr[-1] < -0.55:
                    self.close(size=1000)
                    
                 elif self.dyn_corr[-1] <= -0.55:
                    self.close(size=1000)
                    self.sell(size=1000)
            elif self.position.size<0:
                  if self.dyn_corr[-1]>0 and self.dyn_corr[-1]<0.4  :
                     self.close(size=1000)
                  elif self.dyn_corr[-1] >= 0.4:
                      self.close(size=1000)
                      self.buy(size=1000)
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)
                  
################################################### strategy n. 3 ###################################################
                
class trend_base(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.dollar_eur_trend = data.dollar_eur_trend
    def next(self):
      if self.dollar_eur_trend[-1]>20:
        
      
            if not self.position:
                if self.dyn_corr[-1] < -0.35:
                    self.sell(size=10)
                if self.dyn_corr[-1]>0.35:
                    self.buy(size=10)
     
    
            else:
                if self.position.size >0:
                    
                     if self.dyn_corr[-1] < 0.04 and self.dyn_corr[-1] < -0.35:
                        self.close(size=10)
                        
                     elif self.dyn_corr[-1] <= 0:
                        self.close(size=10)
                        self.sell(size=10)
                elif self.position.size<0:
                      if self.dyn_corr[-1]>0 and self.dyn_corr[-1]<0.35  :
                         self.close(size=10)
                      elif self.dyn_corr[-1] >= 0.35:
                          self.close(size=10)
                          self.buy(size=10)
      portvalue = cerebro.broker.getvalue()

      print('Portfolio Value: ${}'.format(portvalue))
      print(self.position.size)             

################################################### strategy n. 3 ###################################################
                
class buy(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
    
    def next(self):
        if not self.position:
            if self.dyn_corr[-1]>0.40:
                self.buy(size=10)
 

        else:
            if self.position.size >0:
                
                 if self.dyn_corr[-1] < -0.1:
                    self.close(size=10)
                    
                 
            
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)        

################################################### strategy n. 4 ###################################################
                
class pos_trend(bt.Strategy):
    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.diff_wei_trend = data.diff_wei_trend
    def next(self):
         if self.diff_wei_trend>0: 
            if not self.position:
                if self.dyn_corr[-1] < -0.55:
                    self.sell(size=10)
                if self.dyn_corr[-1]>0.40:
                    self.buy(size=10)
     
    
            else:
                if self.position.size >0:
                    
                     if self.dyn_corr[-1] < -0.1 and self.dyn_corr[-1] < -0.55:
                        self.close(size=10)
                        
                     elif self.dyn_corr[-1] <= -0.55:
                        self.close(size=10)
                        self.sell(size=10)
                elif self.position.size<0:
                      if self.dyn_corr[-1]>0 and self.dyn_corr[-1]<0.4  :
                         self.close(size=10)
                      elif self.dyn_corr[-1] >= 0.4:
                          self.close(size=10)
                          self.buy(size=10)
            portvalue = cerebro.broker.getvalue()
        
            print('Portfolio Value: ${}'.format(portvalue))
            print(self.position.size)

 ################################################### strategy n. 5 ###################################################
                
class bip_trend(bt.Strategy):
    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.diff_wei_trend = data.diff_wei_trend
    def next(self):
         if self.diff_wei_trend[-1]>20: 
            if not self.position:
                if self.dyn_corr[-1] < -0.35:
                    self.sell(size=2500)
                if self.dyn_corr[-1]>0.55:
                    self.buy(size=2500)
     
    
            else:
                if self.position.size >0:
                    
                     if self.dyn_corr[-1] < -0.05 and self.dyn_corr[-1] < -0.35:
                        self.close(size=2500)
                        
                     elif self.dyn_corr[-1] <= -0.35:
                        self.close(size=2500)
                        self.sell(size=2500)
                elif self.position.size<0:
                      if self.dyn_corr[-1]>0 and self.dyn_corr[-1]<0.55  :
                         self.close(size=2500)
                      elif self.dyn_corr[-1] >= 0.55:
                          self.close(size=2500)
                          self.buy(size=2500)
         elif self.diff_wei_trend[-1]<-5: 
                        if not self.position:
                            if self.dyn_corr[-1] < -0.35:
                                self.buy(size=2500)
                            if self.dyn_corr[-1]>0.55:
                                self.sell(size=2500)
                 
                
                        else:
                            if self.position.size <0:
                                
                                 if self.dyn_corr[-1] < -0.05 and self.dyn_corr[-1] > -0.35:
                                    self.close(size=2500)
                                    
                                 elif self.dyn_corr[-1] <= -0.35:
                                    self.close(size=2500)
                                    self.buy(size=2500)
                            elif self.position.size>0:
                                  if self.dyn_corr[-1]>0.05 and self.dyn_corr[-1]<0.35  :
                                     self.close(size=2500)
                                  elif self.dyn_corr[-1] >= 0.55:
                                      self.close(size=2500)
                                      self.sell(size=2500)
                        portvalue = cerebro.broker.getvalue()
                    
                        print('Portfolio Value: ${}'.format(portvalue))
                        print(self.position.size)       


 ################################################### strategy n. 6 ###################################################


class quantity(bt.Strategy):
    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.diff_wei_trend = data.diff_wei_trend
        self.abs_var = data.abs_var
    def next(self):
         if self.diff_wei_trend[-1]>20: 
            if not self.position:
                if self.dyn_corr[-1] < -0.35:
                    self.sell(size=3000*self.dyn_corr[-1]*self.abs_var[-1]*50)
                if self.dyn_corr[-1]>0.55:
                    self.buy(size=3000*self.dyn_corr[-1]*self.abs_var[-1]*50)
     
    
            else:
                if self.position.size >0:
                    
                     if self.dyn_corr[-1] < -0.05 and self.dyn_corr[-1] < -0.35:
                        self.close()
                        
                     elif self.dyn_corr[-1] <= -0.35:
                        self.close()
                        self.sell(size=3000*self.dyn_corr[-1]*self.abs_var[-1]*50)
                elif self.position.size<0:
                      if self.dyn_corr[-1]>0 and self.dyn_corr[-1]<0.55  :
                         self.close()
                      elif self.dyn_corr[-1] >= 0.55:
                          self.close()
                          self.buy(size=3000*self.dyn_corr[-1]*self.abs_var[-1]*50)
         elif self.diff_wei_trend[-1]<-5: 
                        if not self.position:
                            if self.dyn_corr[-1] < -0.35:
                                self.buy(size=3000*self.dyn_corr[-1]*self.abs_var[-1]*50)
                            if self.dyn_corr[-1]>0.55:
                                self.sell(size=3000*self.dyn_corr[-1]*self.abs_var[-1]*50)
                 
                
                        else:
                            if self.position.size <0:
                                
                                 if self.dyn_corr[-1] < -0.05 and self.dyn_corr[-1] > -0.35:
                                    self.close()
                                    
                                 elif self.dyn_corr[-1] <= -0.35:
                                    self.close()
                                    self.buy(size=2500*self.dyn_corr[-1]*self.abs_var[-1]*50)
                            elif self.position.size>0:
                                  if self.dyn_corr[-1]>0.05 and self.dyn_corr[-1]<0.35  :
                                     self.close()
                                  elif self.dyn_corr[-1] >= 0.55:
                                      self.close()
                                      self.sell(size=2500*self.dyn_corr[-1]*self.abs_var[-1]*50)
                        portvalue = cerebro.broker.getvalue()
                    
                        print('Portfolio Value: ${}'.format(portvalue))
                        print(self.position.size)               
startcash = 3000
cerebro = bt.Cerebro()
cerebro.addstrategy(quantity)###########################Add strategy

#Get S&P500 from CSV
data = GenericCSV_PE(
    dataname= r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\dataset2_dollar_eur.csv',
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
#     volume = 9,
    dyn_corr =23,
    dollar_eur_trend=24,
    diff_wei_trend = 25,
    abs_var = 26
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
result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\result.csv")


import pyfolio as pf
pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions)
   # round_trips=True)
   
   
transactions.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 5\dataset\transactions.csv")
