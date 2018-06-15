import numpy as np
import pandas as pd

import protos_edge as pe

def initPortfolio(prices, txvol, param,strategy):
    if(strategy == "nvx"): signals = getSignals("nvx",prices.iloc[:1,:],txvol.iloc[:1,:],param)
    portfolio = dict({'balance':100,'positions':(signals*0)})
    
    return portfolio
    


def updatePositions(portfolio, targetAlloc,prices, spread):
    
    deltaTrades = (targetAlloc*portfolio['balance']/prices.iloc[prices.shape[0]-1,:] - portfolio['positions']).fillna(0)
    
    # Execute Target Allocation: Put Target Quantities of Tickers into the Portfolio
    portfolio['positions'] = targetAlloc*portfolio['balance']/prices.iloc[prices.shape[0]-1,:]
    
    # Subtract Fees in the form of Spread above/below market price from Balance
    portfolio['balance'] -= abs((prices.iloc[prices.shape[0]-1,:]*deltaTrades*spread).sum())
     
    return portfolio
    


def updateBalance(portfolio, prices):
    
    portfolio['balance'] += (portfolio['positions']*(prices.iloc[prices.shape[0]-1]-prices.iloc[prices.shape[0]-2])).sum()
    
    return portfolio


def VolNorming(signals,prices):
    
    returns = prices.pct_change()
    ewma_init = 30
    ewma = signals*0
    if(returns.shape[0] >= ewma_init):
        ewma = signals*(returns.iloc[:ewma_init,:].std(axis=0)**2)
        for i in range(ewma_init+1,returns.shape[0]):
            ewma = 0.94*ewma.squeeze() + 0.06*((signals*returns.iloc[i,:].rename())**2).T.squeeze()
        ewma = (ewma.sum()/ewma).replace(np.inf, np.nan)
        ewma = ewma/ewma.sum()
    weights = ewma*signals
    
    return weights


def equalWeight(signals):
    active_tickers = signals.astype(bool).sum()
    return signals/active_tickers

def RiskManagement(signals,portfolio,prices):

    # Weight Signals for Exponential Volatility
    #targetAlloc = VolNorming(signals,prices)
    targetAlloc = equalWeight(signals)
    
    return targetAlloc


def getSignals(strategy,prices,data,n):
    
    if(strategy == "nvx"): signals =  nvx(prices,data,n)
    
    return signals

    
def nvx(prices,nvx,n):    
    if(n[0] == "largest"): selection = nvx.iloc[nvx.shape[0]-1,:].nlargest(n[1])
    if(n[0] == "smallest"): selection = nvx.iloc[nvx.shape[0]-1,:].nsmallest(n[1])
    if(n[0] == "long-short"): 
        long_selection = nvx.iloc[nvx.shape[0]-1,:].nlargest(n[1])
        short_selection = nvx.iloc[nvx.shape[0]-1,:].nsmallest(n[1])
        long_signals = prices.iloc[prices.shape[0]-1,:]*long_selection
        short_signals = prices.iloc[prices.shape[0]-1,:]*short_selection
        signals = long_signals.notnull().astype('int') - short_signals.notnull().astype('int')
    if(n[0] == "short-long"): 
        long_selection = nvx.iloc[nvx.shape[0]-1,:].nsmallest(n[1])
        short_selection = nvx.iloc[nvx.shape[0]-1,:].nlargest(n[1])
        long_signals = prices.iloc[prices.shape[0]-1,:]*long_selection
        short_signals = prices.iloc[prices.shape[0]-1,:]*short_selection
        signals = long_signals.notnull().astype('int') - short_signals.notnull().astype('int')  
    if((n[0] != "long-short") & (n[0] != "short-long")):    
        signals = prices.iloc[prices.shape[0]-1,:]*selection
        signals = signals.notnull().astype('int')
    return signals


def loadData(nvx):
    # Data
    tickers_tx=['btc', 'eth', 'xrp', 'ltc', 'dash', 'xem', 'etc', 'zec', 'pivx',
             'gnt', 'dcr', 'dgb', 'doge', 'xvg']
    
    tickers = ['bitcoin','ethereum','ripple','litecoin','dash','nem', 'ethereum-classic',
               'zcash','pivx','golem-network-tokens','decred','digibyte','dogecoin',
               'verge']
    
    start = '2017-01-01'
    end = '2018-06-13'
    
    if(nvx == "nvt"):
        volume = pe.Data_Selected(start,end,frequency=1,tickers=tickers_tx)
        vol = volume.load_data(table="txvolume")
        vol = volume.clean_data(vol)
        vol.columns = tickers
        
    if(nvx == "nvv"):
        volume = pe.Data_Selected(start,end,frequency=1,tickers=tickers)
        vol = volume.load_data(table="volume")
        vol = volume.clean_data(vol)
        vol.columns = tickers        
    
    
    data = pe.Data_Selected(start,end,frequency=1,tickers=tickers)
    prices = data.load_data(table="close")
    prices = data.clean_data(prices)
    
    mcap = data.load_data(table="mcap")
    mcap = data.clean_data(mcap)
    
    return vol, prices, mcap