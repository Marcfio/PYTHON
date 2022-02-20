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
import pandas as pd

################class for importing data########################################àààà
class GenericCSV_PE(btfeeds.GenericCSVData):

    # Add a 'pe' line to the inherited ones from the base class
    lines = ('dyn_corr','trump_trend',)
    
    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('dyn_corr', 8),('trump_trend',9),)


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
            if self.dyn_corr[-1] < -0.5:
                self.sell(size=1)
            if self.dyn_corr[-1]>0.3:
                self.buy(size=1)
 

        else:
            if self.position.size >0:
                
                 if self.dyn_corr[-1] < -0.2:
                    self.close(size=1)
                    
                
            elif self.position.size<0:
                  if self.dyn_corr[-1]>0.2  :
                    self.close(size=1)
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)

################################################### strategy n. 3 ###################################################
        
class lev(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.trump_trend = data.trump_trend
        self.volume = data.volume
    def next(self):
        if not self.position:
         #   if self.trump_trend[-1]>0:
                if self.dyn_corr[-1] < -0.1:
                    self.sell(size=1)
                if self.dyn_corr[-1]>0.1:
                    self.buy(size=1)
 
        else:
           # if self.trump_trend[-1]>0:
                if self.position.size >0:

                     if self.dyn_corr[-1] < -0.05:
                        self.close(size=1)


                elif self.position.size<0:
                      if self.dyn_corr[-1]>0.05  :
                        self.close(size=1)
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)     
        
################################################### strategy n. 4 ###################################################
        
class everytimeisagoodtime(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.trump_trend = data.trump_trend
        self.volume = data.volume
    def next(self):
        if not self.position:
            
                if self.dyn_corr[-1] < -0.1:
                    self.sell(size=1)
                if self.dyn_corr[-1]>0.1:
                    self.buy(size=1)
 
        else:
            
                if self.position.size >0:

                     if self.dyn_corr[-1] < -0.08:
                        self.close(size=1)
                        self.sell(size=1)

                elif self.position.size<0:
                      if self.dyn_corr[-1]> 0.05  :
                        self.close(size=1)
                        self.buy(size=1)
        portvalue = cerebro.broker.getvalue()

        print('Portfolio Value: ${}'.format(portvalue))
        print(self.position.size)     
                
 #####################################trategt n. 5###################à       
class trend_base(bt.Strategy):

    def __init__(self):
        self.dyn_corr = data.dyn_corr
        self.trump_trend = data.trump_trend
        self.volume = data.volume
        self.gen_trend = data.trump_trend
    def next(self):
         if self.gen_trend>0:  
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
startcash = 3000
cerebro = bt.Cerebro()
cerebro.addstrategy(everytimeisagoodtime)

#Get S&P500 from CSV
data = GenericCSV_PE(
    dataname= r'C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dataset.csv',
    fromdate=datetime(2018, 1, 1),
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
    dyn_corr = 14,
    trump_trend = 13 
        )

cerebro.adddata(data)

cerebro.broker.setcash(startcash)

cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
results = cerebro.run()
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
result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\result_S&P500_trade.csv")



import pyfolio as pf
pf.create_full_tear_sheet(
    returns,
    positions=positions,
    transactions=transactions)
   # round_trips=True)
   
result = pd.concat([positions,returns],axis=1, join = 'outer')
result['portfolio']= result.dataset + result.cash
result.to_csv(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\result_S&P500_trade.csv")
result.to_excel(r"C:\Users\MARCOFIORAVANTIPC\Google Drive\MSC FINANCE AND BANKING\thesis_2\apollo 1\dataset\dataset_graph\result_S&P500_trade.xlsx", index = False)


##############analyze strategy

