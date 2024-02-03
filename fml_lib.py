import numpy as np
import pandas as pd

def get_tick_bars( history, tick_count:int):
    tick_bars = history.iloc[::tick_count,:]
    
    tick_open = history.groupby(np.arange(len(history))// tick_count)[['Bid price']].min()
    tick_close = history.groupby(np.arange(len(history))// tick_count)[['Bid price']].max()
    tick_bars['open'] = tick_open.to_numpy()
    tick_bars['close'] = tick_close.to_numpy()
    return tick_bars

def get_volume_bars( history,  volume_count:int):
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
            new_k = (h*(close-open)) + k_list[index-1]
        k_list.append(new_k)
    bars['K'] = k_list
    bars['returns'] = bars['K'] / (bars['K'].shift(1,fill_value=1))
    return bars

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