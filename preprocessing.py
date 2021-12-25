import numpy as np 
import pandas as pd
from pathlib import Path
from datetime import datetime


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
    
    def test_null(self, f_name):
        '''
        f_name : test the file name
        '''
        path = self.path/Path(f_name)
        df = pd.read_csv(path)
        print(f'Now check if  {f_name} has null:')
        for c in df.columns:
            print(f"    null rate of column——{c} :  {df.loc[:,c].isnull().sum()/len(df)}")