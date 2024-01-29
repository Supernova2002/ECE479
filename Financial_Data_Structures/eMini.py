import numpy as np
import yfinance as yf
import pandas as pd
import sys
import os

def get_tick_bars( history, tick_count:int):
    tick_bars = history.iloc[::tick_count,:]
    tick_open = history.groupby(np.arange(len(history))// tick_count)[['Bid price']].max()
    tick_close = history.groupby(np.arange(len(history))// tick_count)[['Bid price']].min()
    tick_bars['open'] = tick_open
    tick_bars['close'] = tick_close
    return tick_bars

def get_volume_bars( history,  volume_count:int):
    volume_sum = 0
    volume_bars = pd.DataFrame(columns=history.columns)
    volume_open = []
    volume_close = []
    local_min = 0
    local_max = 0
    for index,values in enumerate(zip(history['Bid volume'], history['Bid price'])):
        #print(index)
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

def get_dollar_bars( history,  dollar_count:int):
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
    history = history[(history['Timestamp'] >'20230702 23:59:59:999' )  & (history['Timestamp'] < '20230731 00:00:00:000')]
    

    # form tick, volume, and dollar bars

    # set tick count for tick bar to 500
    tick_count = 500
    if not os.path.exists("./july_2023_tick_bars.csv"):
        tick_bars = get_tick_bars(history,tick_count)
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
        volume_bars.to_csv("./july_2023_volume_bars.csv",index=False)
    else:
        volume_bars = pd.read_csv("./july_2023_volume_bars.csv")
    print(volume_bars)
    volume_bars['Timestamp'] = pd.to_datetime(volume_bars['Timestamp'],format = '%Y%m%d %H:%M:%S:%f')
    weekly_volume_bars = [g for n,g in volume_bars.groupby(pd.Grouper(key="Timestamp",freq="W"))]
    weekly_volume_bars_count = []
    for week in weekly_volume_bars:
        weekly_volume_bars_count.append(len(week))
    
    dollar_count = 100000
    if not os.path.exists("./july_2023_dollar_bars.csv"):
        dollar_bars = get_dollar_bars(history, dollar_count)
        dollar_bars.to_csv("./july_2023_dollar_bars.csv",index=False)
    else:
        dollar_bars = pd.read_csv("./july_2023_dollar_bars.csv")
    dollar_bars['Timestamp'] = pd.to_datetime(dollar_bars['Timestamp'],format = '%Y%m%d %H:%M:%S:%f')
    weekly_dollar_bars = [g for n,g in dollar_bars.groupby(pd.Grouper(key="Timestamp",freq="W"))]
    weekly_dollar_bars_count = []
    for week in weekly_dollar_bars:
        weekly_dollar_bars_count.append(len(week))

    #weekly tick count for a month
    print(weekly_tick_bars_count)
    print(weekly_volume_bars_count)
    print(weekly_dollar_bars_count)
    #print(dollar_bars)