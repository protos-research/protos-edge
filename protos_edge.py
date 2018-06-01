import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import sqlalchemy as sql
import math

from base_classes import Data, Portfolio, Strategy, Backtest


###############################################################################

class Data_Selected(Data):
    
    # Get Data Set with closing prices for bitcoin, bitcoin-cash, ethereum, litecoin
    # ... ripple
    
    def __init__(self,start,end,frequency,tickers):
        self.start = start
        self.end = end
        self.frequency = frequency
        self.tickers = tickers
        
    def load_data(self):
        tickers = ['Date'] + self.tickers
        ticker_str = ', '.join("`{}`".format(ticker) for ticker in tickers)

        engine = sql.create_engine('mysql+pymysql://protos-github:protos-github@google-sheet-data.cfyqhzfdz93r.eu-west-1.rds.amazonaws.com:3306/protos')

        prices = pd.read_sql("Select " + str(ticker_str) + " From close", con=engine)
        
        return prices
    
    def clean_data(self, data):
        date_filter = (data['Date'] >= self.start) & (data['Date'] <= self.end)
        price = data[date_filter]
        # frequency_filter = data['Date'] == ...
        # price = price[frequency_filter]
        price.set_index('Date', inplace=True)
        price.index = pd.to_datetime(price.index)
        price.fillna('NaN')
        price = price.apply(pd.to_numeric, errors='coerce')
        return price
        

class Trend_Following(Strategy):   
    
    def __init__(self, max_lookback, weights, 
                 normalize_vol, long_only, short_only):
        self.max_lookback = max_lookback
        self.weights = weights
        self.normalize_vol = normalize_vol
        self.long_only = long_only
        self.short_only = short_only

    def generate_signals(self, prices):
        last_row = prices.shape[0]-1
        lb1 = int(self.max_lookback/3)
        lb2 = int(2*self.max_lookback/3)
            # As soon as singals are fully calculated
        if(last_row >= self.max_lookback):
            l_mask_1 = prices.iloc[last_row,:]>=prices.iloc[last_row-lb1,:]
            l_mask_1 = l_mask_1*self.weights[0]
            l_mask_1.mask(l_mask_1==0,other=(-self.weights[0]), inplace=True)
                    
            l_mask_2 = prices.iloc[last_row,:]>=prices.iloc[last_row-lb2,:]
            l_mask_2 = l_mask_2*self.weights[1]
            l_mask_2.mask(l_mask_2==0,other=(-self.weights[1]), inplace=True)
    
            l_mask_3 = prices.iloc[last_row,:]>=prices.iloc[last_row-self.max_lookback,:]
            l_mask_3 = l_mask_3*self.weights[2]
            l_mask_3.mask(l_mask_3==False,other=(-self.weights[2]), inplace=True)
    
            #### Short Masks
            
            s_mask_1 = prices.iloc[last_row,:]<prices.iloc[last_row-lb1,:]
            s_mask_1 = s_mask_1*(-self.weights[0])
            s_mask_1.mask(s_mask_1==0,other=(self.weights[0]), inplace=True)
            
            s_mask_2 = prices.iloc[last_row,:]<prices.iloc[last_row-lb2,:]
            s_mask_2 = s_mask_2*(-self.weights[1])
            s_mask_2.mask(s_mask_2==0,other=(self.weights[1]), inplace=True)
    
            s_mask_3 = prices.iloc[last_row,:]<prices.iloc[last_row-self.max_lookback,:]
            s_mask_3 = s_mask_3*(-self.weights[2])
            s_mask_3.mask(s_mask_3==0,other=(self.weights[2]), inplace=True)
    
            for index, i in enumerate(prices.iloc[last_row-self.max_lookback,:]):
                if(math.isnan(i)): 
                    l_mask_1[index] = np.NAN
                    l_mask_2[index] = np.NAN
                    l_mask_3[index] = np.NAN
                    s_mask_1[index] = np.NAN
                    s_mask_2[index] = np.NAN
                    s_mask_3[index] = np.NAN
                          
            # Long-Only or Long-Short   
            if(self.long_only):
                mask = l_mask_1 + l_mask_2 + l_mask_3
                mask.mask(mask < 0, other=0, inplace=True)
            elif(self.short_only):
                mask = s_mask_1 +s_mask_2 + s_mask_3
                mask.mask(mask > 0, other=0, inplace=True)
            else:
                mask = l_mask_1 + l_mask_2 + l_mask_3 

        else:
            mask = prices.iloc[last_row,:]
            mask = (mask*0).fillna(0)
    
        ewma_ann = [0,0,0,0,0]
        # Normalize for Volatility as well:
        vol_lb = 90
        if(last_row+1 >= vol_lb):
            if(self.normalize_vol):
                returns = prices.pct_change().replace(np.inf, np.nan)
                ewma0 = returns.iloc[:vol_lb,:].std(axis=0)**2
                if(last_row>0):
                    for i in range(vol_lb,last_row+1):#returns.shape[0]-vol_lb .... vol_lb+i
                        ewma0 = 0.94*ewma0.squeeze() + 0.06*((returns.iloc[i,:].rename())**2).T.squeeze() 
                ewma_ann = np.sqrt(ewma0)*np.sqrt(365)
                ewma = ewma_ann.sum()/ewma_ann
                ewma_norm = ewma/ewma.sum()
                mask = mask*ewma_norm

        # Normalize the mask - max single position risk = 1/(nr of tickers active)
        if(self.normalize_vol): mask_norm = mask  
        else: mask_norm = mask/mask.count()
        #Replace NaN with 0 
        mask_norm = mask_norm.fillna(0)
        return mask_norm
 
       
class Daily_Portfolio(Portfolio):
    
    def __init__(self, init_balance = 0, balance=[], positions=[], trading=[],fees=0):
        self.positions = positions
        self.init_balance = init_balance
        self.balance = balance  
        self.fees = fees
        self.trading = trading
        
        
class Daily_Backtest(Backtest):
    
    def __init__(self, rebalance_period, spread, fees):
        self.rebalance_period = rebalance_period
        self.spread = spread
        self.fees = fees
    
    def run_backtest(self, data, portfolio,strategy):
        balance = portfolio.init_balance
        for i in range(1,data.shape[0]):  
        ### What happened to our portfolio during the timestep?
        # Add returns to balance, if we had a non-empty portfolio allocation
            if(i > 1):
                # update current balance only when portfolio has allocations
                # for the first days, there are no trend-signals == no allocation
                if(abs(portfolio.positions[len(portfolio.positions)-1]).sum() != 0):
                    # add returns of each ticker for this timestep
                    # quantity of ticker * price_delta (quantity is neg for short pos)
                    balance += (portfolio.positions[len(portfolio.positions)-1]*(data.iloc[i-1,:]-data.iloc[i-2,:])).sum()
            
            ### How should we react to new prices?
                
            # get new weights
            allocation = strategy.generate_signals(data.iloc[0:i,:])
                
            # calculate target allocation
            t_alloc = allocation*balance
            #tweights.append(allocation)
            # calculate target quantity allocation        
            q_alloc = (t_alloc/data.iloc[i-1,:]).fillna(0)
            # change division by zero (no prices available, etc.) to 0
            q_alloc = q_alloc.replace(np.inf, 0)
            
            
            # change quantity allocation of our portfolio 
            # i%7 == 0 every seven days! On all other days, portfolio allocation stays unchanged
            if(i == 1): portfolio.positions.append(q_alloc)
            if((i%self.rebalance_period == 0) & (i != 1)):
                # Append new allocation to portfolio every x=rebalancing_period days                      
                portfolio.positions.append(q_alloc)
                # Subtract transaction fees and market spread
                trades = portfolio.positions[len(portfolio.positions)-1]-portfolio.positions[len(portfolio.positions)-2]
                portfolio.trading.append(trades)
                balance -= (abs(portfolio.positions[len(portfolio.positions)-1])*data.iloc[i-1,:]*self.spread).sum()
                #balance -= fees*trading.count()
       
            # add current days new balance (calculated above) as soon as signals start to come in (i > lookback-period for trend signals)
            if(i >= strategy.max_lookback):
                portfolio.balance.append(balance)
            
        return portfolio.balance
        
        
    def collect_statistics(self, portfolio_balance):
        portfolio_balance = pd.DataFrame(portfolio_balance)
        returns = portfolio_balance.pct_change()
        sharpe = returns.mean()/(returns.std())*np.sqrt(365)
        mean = returns.mean()*365
        vol = returns.std()*np.sqrt(365)
        gain_to_pain = returns.sum()/abs(returns[returns < 0].sum())
        
        print("Expected Returns: " + str(mean.values))
        print("Volatility: " + str(vol.values))
        print("-------------------------------------")
        print("Sharpe Ratio: " + str(sharpe.values))
        print("Gain to Pain: " + str(gain_to_pain.values))
        print("-------------------------------------")
        #print(returns.describe())
        print("-------------------------------------")
        print("Final Balance: " + str(portfolio_balance.iloc[portfolio_balance.shape[0]-1].values))
        portfolio_balance.plot()


    

    

    
