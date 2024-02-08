import fml_lib
import pandas as pd
import sys

if __name__ == "__main__":

    # chapter 3 exercise 1
    dollar_bars = pd.read_csv("./july_2023_dollar_bars.csv")
    dollar_bars['Timestamp'] = pd.to_datetime(dollar_bars['Timestamp'],format = '%Y%m%d %H:%M:%S:%f')
    dollar_bars.set_index(["Timestamp"],inplace=True)
    close = dollar_bars['close']
    daily_volatility = fml_lib.getDailyVol(dollar_bars.close,span0=100)
    t_events = fml_lib.getTEvents(dollar_bars.close, daily_volatility)
    numDays = 1
    t1=close.index.searchsorted(t_events+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=pd.Series(close.index[t1],index=t_events[:t1.shape[0]]) # NaNs at end
    min_ret = 0.00001
    trgt = pd.Series(0.00003, index=t_events)
    events = fml_lib.getEvents(close, t_events, 1,trgt ,min_ret,t1)
    bin = fml_lib.getBins(events, close)
    print(bin)