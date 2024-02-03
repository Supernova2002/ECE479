import numpy as np
import yfinance as yf
import pandas as pd
import sys
import os

def get_tick_bars( history, tick_count:int):
   # history['avg_price'] = (history['Bid price'] + history['Ask price'])/2
    tick_bars = history.iloc[::tick_count,:]
    
    tick_open = history.groupby(np.arange(len(history))// tick_count)[['Bid price']].min()
    tick_close = history.groupby(np.arange(len(history))// tick_count)[['Bid price']].max()
    tick_bars['open'] = tick_open.to_numpy()
    tick_bars['close'] = tick_close.to_numpy()
    return tick_bars

def get_volume_bars( history,  volume_count:int):
   # history['avg_price'] = (history['Bid price'] + history['Ask price'])/2
    volume_sum = 0
    volume_bars = pd.DataFrame(columns=history.columns)
    volume_open = []
    volume_close = []
    local_min = 0
    local_max = 0
    for index,values in enumerate(zip(history['Bid volume'], history['Bid price'])):
        volume = values[0]
        price = values[1]
        volume_sum = volume_sum + volume
        if local_min == 0: 
            local_min = price
        if price< local_min:
            local_min = price
        if price> local_max:
            local_max = price
        if volume_sum>volume_count:
            volume_bars = pd.concat([volume_bars,(history.iloc[[index]])],axis=0, ignore_index=True)
            volume_open.append(local_min)
            volume_close.append(local_max)
            volume_sum = 0
    volume_bars['open'] = volume_open
    volume_bars['close'] = volume_close
    return volume_bars


def get_roll(bars):
    # phi is row.close - row.open
    # h is  K / next open, which is just last K /row.open
    #print(bars['close'])

    k_list = []
    for index, values in enumerate(zip(bars['open'], bars['close'])):
        open = values[0]
        close = values[1]
        if index == 0:
            new_k = 1
        else:
            h = k_list[index-1] / open
            transaction_cost = 0.0001
            #rebalance_cost = (h*close + (k_list[index]/open[index+1]) * open[index+1])*transaction_cost
            new_k = (h*(close-open)) + k_list[index-1]
        k_list.append(new_k)
    bars['K'] = k_list
    bars['returns'] = bars['K'] / (bars['K'].shift(1,fill_value=1))
    #bars['K'] = bars.apply(lambda row: ((row['close']-row['open'])/row['open']) + row['K'].shift(1,fill_value = 1), axis=1) 
    return bars

def get_dollar_bars( history,  dollar_count:int):
    #history['avg_price'] = (history['Bid price'] + history['Ask price'])/2
    dollar_sum = 0
    dollar_bars = pd.DataFrame(columns=history.columns)
    dollar_open = []
    dollar_close = []
    local_min = 0
    local_max = 0
    for index, i in enumerate(history["Bid price"]):
        dollar_sum = dollar_sum + i
        if local_min == 0: 
            local_min = i
        if i< local_min:
            local_min = i
        if i> local_max:
            local_max = i
        if dollar_sum>dollar_count:
            dollar_open.append(local_min)
            dollar_close.append(local_max)
            dollar_bars = pd.concat([dollar_bars,(history.iloc[[index]])],axis=0, ignore_index=True)
            dollar_sum = 0
    dollar_bars["open"] = dollar_open
    dollar_bars["close"] = dollar_close
    return dollar_bars


if __name__ == "__main__":
    history = pd.read_csv("../USA500IDXUSD.csv") 
    full_history = history
    history = history[(history['Timestamp'] >'20230702 23:59:59:999' )  & (history['Timestamp'] < '20230731 00:00:00:000')]
    

    # form tick, volume, and dollar bars

    # set tick count for tick bar to 500
    tick_count = 500
    if not os.path.exists("./july_2023_tick_bars.csv"):
        tick_bars = get_tick_bars(history,tick_count)
        tick_bars = get_roll(tick_bars)
        tick_bars.to_csv("./july_2023_tick_bars.csv",index=False)
    else:
        tick_bars = pd.read_csv("./july_2023_tick_bars.csv")
    tick_bars['Timestamp'] = pd.to_datetime(tick_bars['Timestamp'],format = '%Y%m%d %H:%M:%S:%f')
    #print(tick_bars)
    weekly_tick_bars = [g for n,g in tick_bars.groupby(pd.Grouper(key="Timestamp",freq="W"))]
    weekly_tick_bars_count = []
    for week in weekly_tick_bars:
        weekly_tick_bars_count.append(len(week))
    
    # volume bar time
    volume_count = 0.01
    if not os.path.exists("./july_2023_volume_bars.csv"):
        volume_bars = get_volume_bars(history, volume_count)
        volume_bars = get_roll(volume_bars)
        volume_bars.to_csv("./july_2023_volume_bars.csv",index=False)
    else:
        volume_bars = pd.read_csv("./july_2023_volume_bars.csv")
    volume_bars['Timestamp'] = pd.to_datetime(volume_bars['Timestamp'],format = '%Y%m%d %H:%M:%S:%f')
    weekly_volume_bars = [g for n,g in volume_bars.groupby(pd.Grouper(key="Timestamp",freq="W"))]
    weekly_volume_bars_count = []
    for week in weekly_volume_bars:
        weekly_volume_bars_count.append(len(week))
    
    dollar_count = 100000
    if not os.path.exists("./july_2023_dollar_bars.csv"):
        dollar_bars = get_dollar_bars(history, dollar_count)
        dollar_bars = get_roll(dollar_bars)
        dollar_bars.to_csv("./july_2023_dollar_bars.csv",index=False)
    else:
        dollar_bars = pd.read_csv("./july_2023_dollar_bars.csv")
    dollar_bars['Timestamp'] = pd.to_datetime(dollar_bars['Timestamp'],format = '%Y%m%d %H:%M:%S:%f')
    sys.exit(1)
    weekly_dollar_bars = [g for n,g in dollar_bars.groupby(pd.Grouper(key="Timestamp",freq="W"))]
    weekly_dollar_bars_count = []
    for week in weekly_dollar_bars:
        weekly_dollar_bars_count.append(len(week))
    
    #weekly tick count for a month
    print(weekly_tick_bars_count)
    print(weekly_volume_bars_count)
    print(weekly_dollar_bars_count)

    #serial correlation for each
    #print(get_roll(tick_bars))
    print(pd.Series.autocorr(get_roll(tick_bars)['returns']))
    print(pd.Series.autocorr(get_roll(volume_bars)['returns']))
    print(pd.Series.autocorr(get_roll(dollar_bars)['returns']))

    # tick bars have by the far the lowest method of serial correlation


    #Partitioning into monthly subsets now
    #tick data first
    full_history['Timestamp'] = pd.to_datetime(full_history['Timestamp'],format = '%Y%m%d %H:%M:%S:%f')
    monthly_history = [g for n,g in full_history.groupby(pd.Grouper(key="Timestamp",freq="M"))]
    tick_monthly_variance = []
    volume_monthly_variance = []
    dollars_monthly_variance = []
    print(len(monthly_history))
    for month in monthly_history:
        monthly_ticks = get_tick_bars(month, tick_count)
        monthly_tick_roll = get_roll(monthly_ticks)
        tick_monthly_variance.append(pd.Series.autocorr(monthly_tick_roll['returns']))
        monthly_volume = get_volume_bars(month,volume_count)
        monthly_volume_roll = get_roll(monthly_volume)
        volume_monthly_variance.append(pd.Series.autocorr(monthly_volume_roll['returns']))
        monthly_dollars = get_dollar_bars(month, dollar_count)
        monthly_dollar_roll = get_roll(monthly_dollars)
        dollars_monthly_variance.append(pd.Series.autocorr(monthly_dollar_roll['returns']))
    tick_month_to_month_variance = pd.Series.autocorr(pd.Series(tick_monthly_variance))
    volume_month_to_month_variance = pd.Series.autocorr(pd.Series(volume_monthly_variance))
    dollars_month_to_month_variance = pd.Series.autocorr(pd.Series(dollars_monthly_variance))
    print("Tick month to month variance is : " + tick_month_to_month_variance)
    print("Volume month to month variance is : " + volume_month_to_month_variance)
    print("Dollar month to month variance is : " + dollars_month_to_month_variance)