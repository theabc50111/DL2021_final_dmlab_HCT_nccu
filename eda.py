import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from pylab import rcParams


class EDA:
    def __init__(self, obj_ticker, price_type):
        self.obj_ticker = obj_ticker
        self.price_type = price_type
        
        
    def draw_ori_price(self,price_df):
        plt.figure(figsize=(10,6))
        plt.grid(True)
        plt.xlabel('Dates', fontsize=14)
        plt.ylabel(f'{self.price_type} Prices', fontsize=14)
        plt.plot(price_df[self.price_type])
        plt.title(f'{self.obj_ticker} closing price', fontsize=19)
        plt.show()
   

    def test_stationarity(self, price_df):
        '''
        adfuller:
            adf （float）: 测试统计
            pvalue （float） : MacKinnon基于MacKinnon的近似p值（1994年，2010年）
            usedlag （int）: 使用的滞后数量
            nobs（ int）: 用于ADF回归的观察数和临界值的计算
            critical values（dict）: 测试统计数据的临界值为1％，5％和10％。基于MacKinnon（2010）
            icbest（float）: 如果autolag不是None，则最大化信息标准。
            resstore （ResultStore，可选）: 一个虚拟类，其结果作为属性附加
            -------------------------------------------------------------------------------
            如何确定该序列能否平稳呢？主要看：
                1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设
                以股票APA，adf结果为-1.42， 大于三个level的统计值。所以是不平稳的，需要进行一阶差分后，再进行检验。
                P-value是否非常接近0，接近0，则是平稳的，否则，不平稳。
            ref:
                - https://blog.csdn.net/weixin_42746776/article/details/103723615
                - https://www.itbook5.com/2019/08/11560/
        '''
        
        timeseries = price_df[self.price_type]
        #Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        rolemean50 = timeseries.rolling(50).mean()
        rolstd50 = timeseries.rolling(50).std()
        #Plot rolling statistics:
        plt.figure(figsize=(10,6))
        plt.plot(timeseries, color='blue',label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='green', label = 'Rolling Std')
        plt.plot(rolemean50, color='magenta', marker='o', label='Rolling Mean 50')
        plt.plot(rolstd50, color='purple', marker='x',label = 'Rolling Std 50')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.show(block=False)

        print("Results of dickey fuller test")
        adft = adfuller(timeseries,autolag='AIC')
        output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used']) 
        for key,values in adft[4].items():
            output[f'critical value ({key})'] =  values
        print(output)

        print('*='*100)
        if (adft[0] > adft[4]['10%']) or (adft[0] > adft[4]['5%']) or (adft[0] > adft[4]['1%']):
            print(f'{self.obj_ticker}\'s {timeseries.name} is not stationary')
        else:
            print(f'{self.obj_ticker}\'s {timeseries.name} is stationary')
        print('*='*100)
        

    def seasonal_decompose(self,price_df, period):
        season_de = seasonal_decompose(price_df[self.price_type], model='multiplicative', period = period)
        fig = plt.figure()  
        fig = season_de.plot()  
        fig.set_size_inches(16, 9)
        fig.text(0.45,1 ,f"period_{period}",fontsize=32)
        plt.show()
