from abc import ABCMeta, abstractmethod


class Data(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def load_data(self):
        raise NotImplementedError("Should implement load_data()!")
        
    @abstractmethod
    def clean_data(self):
        raise NotImplementedError("Should implement clean_data()!")


class Strategy(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def signals(self):
        raise NotImplementedError("Should implement signals()!")

        
class Portfolio(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def update_balance(self):
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def update_positions(self):
        raise NotImplementedError("Should implement backtest_portfolio()!")
        
    @abstractmethod
    def subtract_trading_fees(self):
        raise NotImplementedError("Should implement backtest_portfolio()!")
   

class Backtest(object):
    __metaclass__ = ABCMeta 
    @abstractmethod
    def run_backtest(self):
        raise NotImplementedError("Should implement run_backtest()!")
         
    @abstractmethod
    def collect_statistics(self):
        raise NotImplementedError("Should implement collect_statistics()!")
