import numpy as np
import pandas as pd
import sys
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

def getDailyVol(close,span0=100):
    # daily volatility, reindexed to close
    #breakpoint()
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

def getTEvents(gRaw,h):
    tEvents,sPos,sNeg=[],0,0
    diff=gRaw.diff()
    for i in diff.index[1:]:
        sPos,sNeg=max(0,sPos+diff.loc[i]),min(0,sNeg+diff.loc[i])
        if sNeg<-h.iloc[h.index.get_indexer([i], method='nearest')][0]:
            sNeg=0
            tEvents.append(i)
        elif sPos>h.iloc[h.index.get_indexer([i], method='nearest')][0]:
            sPos=0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)
def applyPtSlOnT1(close,events,ptSl):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    out=events.copy(deep=True)
    sl_list = []
    pt_list = []
    if ptSl[0]>0:
        pt=ptSl[0]*events['trgt']
    else:
        pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0:
        sl=-ptSl[1]*events['trgt']
    else:
        sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events['t1'].fillna(close.index[-1]).items():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1) # path returns
        #print(sl[loc])
        #sys.exit(1)
        #out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
        #print(df0[df0<sl[loc]].index.min())
        sl_list.append(df0[df0<sl[loc]].index.min())
        pt_list.append(df0[df0>pt[loc]].index.min())
        #out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
    out['sl'] = sl_list
    out['pt'] = pt_list
    return out

def getEvents(close,tEvents,ptSl,trgt,minRet,t1=False):
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:
        t1=pd.Series(pd.NaT,index=tEvents)
    #3) form events object, apply stop loss on t1
    side_=pd.Series(1.,index=trgt.index)
    events=pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1).dropna(subset=['trgt'])
    df0=applyPtSlOnT1(close=close,events=events,ptSl=[ptSl,ptSl])
    events=events.drop('side',axis=1)
    events['t1']=df0.dropna(how='all')[['t1','sl','pt']].min(axis=1) # pd.min ignores nan
    
    return events


def getBins(events, close):
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    out = pd.DataFrame(index = events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin'] = np.sign(out['ret'])
    return out