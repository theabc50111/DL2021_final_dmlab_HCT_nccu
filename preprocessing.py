import numpy as np 
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

class GetData:
    '''
    ----------------------------------------------------------------------
      
    provided methods:
        get_ticker_list()
        get_price_data()
        test_null()
        
    ----------------------------------------------------------------------
    '''
    
    def __init__(self, path):
        self.path = path

    def get_ticker_list(self):
        path_price = self.path/Path('FS_sp500_Value.csv')
        temp_df = pd.read_csv(path_price)
        list_ticker = sorted(list(set(temp_df['Ticker'].to_list())))

        return list_ticker
    
    def get_price_data(self, price_type, ticker_list=[], date_index=True, ignore_index=False, only_recent=False, recent_len=1):
        '''
        option:
            price_type: choose:"High", "Low", "Open", "Close", "Adj Close"
            ticker_list: input a list of ticker of stocks
            date_index: True => set date as index
            ignore_index: True => ignore original index
            only_recent: True => get newest price
            recent_len: number：　the number of newest price
        '''
        self.ticker_list = ticker_list
        path_price = self.path/Path('FS_sp500_Value.csv')
        dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
        combined_price = pd.read_csv(path_price, index_col='Date', parse_dates=['Date'], date_parser=dateparse) if date_index else pd.read_csv(path_price)
        df_price = pd.DataFrame({'Ticker':[] ,f'{price_type}': []})

        for symbol in self.ticker_list:
            temp_df = combined_price[combined_price.Ticker.str.fullmatch(symbol)].copy()
            if only_recent == True:
                df_price = pd.concat([df_price,temp_df.loc[temp_df.index[len(temp_df)-recent_len:], ['Ticker', f'{price_type}']]], ignore_index=ignore_index)
            else:
                df_price = pd.concat([df_price,temp_df.loc[:,['Ticker', f'{price_type}']]], ignore_index=ignore_index)
        return df_price
    

class Prep():
    
    @staticmethod
    def deal_null(input_data, fill=False, fill_method='ffill'):
        '''
        input_data : dataframe
        '''
        print(f'Now check if input data has null:')
        if isinstance(input_data, pd.DataFrame):
            for c in input_data.columns:
                print(f"    null rate of column——{c} :  {input_data.loc[:,c].isnull().sum()/len(input_data)}")
        elif isinstance(input_data, pd.Series):
            print(f"    null rate :  {input_data.isnull().sum()/len(input_data)}")
        if fill:
            if isinstance(fill_method, int):
                fill_na_data = input_data.fillna(fill_method)
            elif fill_method=="ffill" or fill_method=="bfill":
                fill_na_data = input_data.fillna(method=fill_method)
                
            return fill_na_data
        else:
            fill_na_data = input_data.dropna(axis=0)
            
            return fill_na_data
            
    @staticmethod
    def log_rolling(timeseries, period):
        '''
        log transform to make data has stationarity
        timeseriess: input a time-seires
        '''
        
        print(f'Now use log-transform on {timeseries.name} to make it has stationarity')
        fig, axes = plt.subplots(1,2)
        fig.set_size_inches(18, 10)
        log_timeseries = np.log(timeseries).rolling(period).mean()
        log_timeseries_std_dev = log_timeseries.rolling(period).std()
        axes[0].plot(timeseries, color='red', label="original")
        axes[0].legend(loc='best')
        axes[0].title.set_text('original timeseries')
        axes[1].plot(log_timeseries_std_dev, color ="blue", label = "std")
        axes[1].plot(log_timeseries, color="green", label = "log-roll")
        axes[1].title.set_text('Log-rolling timeseries')
        axes[1].legend(loc='best')
        plt.show()
        return log_timeseries