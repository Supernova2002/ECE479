import fml_lib
from fml_lib import get_dollar_bars,get_tick_bars,get_volume_bars,get_roll
import pandas as pd
import sys
import os


if __name__ == "__main__":
    history = pd.read_csv("./USA500IDXUSD.csv") 
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