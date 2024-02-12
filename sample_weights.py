import pandas as pd
import fml_lib
import sys
import matplotlib.pyplot as plt
# Chapter 4

def get_co_events(close, t1):
    first = True
    last_index = 0
    overlap_series = []
    for index, row in close.items():
        if first:
            first = False
        else:
            molecule = [last_index, index]
            if molecule[0] in t1.index and molecule[1] in t1.index:
                unique_count = fml_lib.mpNumCoEvents(close, t1, molecule)
                overlap_series.append(unique_count)
        last_index = index
    overlap_df = pd.concat(overlap_series)
    return overlap_df



if __name__ == "__main__":
    # chapter 4 exercise 1
    dollar_bars = pd.read_csv("./july_2023_dollar_bars.csv")
    dollar_bars['Timestamp'] = pd.to_datetime(dollar_bars['Timestamp'],format = '%Y%m%d %H:%M:%S:%f')
    dollar_bars.set_index(["Timestamp"],inplace=True)
    close = dollar_bars['close']
    close_returns = close.diff() /close.shift(1)
    close_returns_std = close_returns.ewm(span=100).std()
    daily_volatility = fml_lib.getDailyVol(dollar_bars.close,span0=100)
    t_events = fml_lib.getTEvents(dollar_bars.close, daily_volatility)
    numDays = 1
    t1=close.index.searchsorted(t_events+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=pd.Series(close.index[t1],index=t_events[:t1.shape[0]]) # NaNs at end
    min_ret = 0.00001
    trgt = pd.Series(0.00003, index=t_events)
    events = fml_lib.getEvents(close, t_events, 1,trgt ,min_ret,t1)
    # getting number of concurrent events
    co_events = get_co_events(close,events)
    close_returns_std = close_returns_std[co_events.index]
    plt.scatter(co_events, close_returns_std)
    plt.xlabel("Number of concurrent observations")
    plt.ylabel("Weighted standard deviation")
    plt.show()
    # Scatterplot makes it clear that the higher the number of labels, the lower the standard deviation


    # exercise 2
    co_events=co_events.loc[~co_events.index.duplicated(keep='last')]
    co_events=co_events.reindex(close.index).fillna(0)
    out = fml_lib.mpSampleTW(events['t1'],co_events, molecule=events.index)
    out = out.dropna()
    print(out)
    out_ar = out.autocorr()
    out.to_csv("out.csv")
    print(out_ar)


    #Autocorr is Nan, meaning that the statistical properties aren't changing at all over time



    

