from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import datetime
import backtrader.feeds as btfeeds






class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[1].datetime.date(1)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        def next(self):
            # Simply log the closing price of the series from the reference
            self.log('Close, %.2f' % self.data.close[0])
    
            if self.dataclose[0] < self.data.close[-1]:
                    # current close less than previous close
    
                    if self.dataclose[-1] < self.data.close[-2]:
                        # previous close less than the previous close
    
                        # BUY, BUY, BUY!!! (with all possible default parameters)
                            self.log('BUY CREATE, %.2f' % self.data.close[0])
                            self.buy()

        
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    cerebro.addstrategy(TestStrategy)

    
    
    data = btfeeds.GenericCSVData(
    dataname='S&P500.csv',

    fromdate=datetime.datetime(2019, 1, 1),
    todate=datetime.datetime(2019, 12, 31),

    nullvalue=0.0,

    dtformat=('%Y-%m-%d'),
  
    datetime=0,
    open=1,
    high=2,
    low=3,

    close=4,
    volume=5,

)

    
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1540000.0)
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())