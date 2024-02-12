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
    events['t1']=df0.dropna(how='all')[['t1','sl','pt']].min(axis=1) # pd.min ignores nan
    events=events.drop('side',axis=1)
    return events


def getBins(events, close):
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    out = pd.DataFrame(index = events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin'] = np.sign(out['ret'])
    return out



def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count
    '''
    molecule = pd.to_datetime(molecule, errors="coerce")
    closeIdx = closeIdx.index
    t1 = t1.drop("trgt", axis=1)
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights\
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1 = t1.dropna()
    t1=t1.loc[t1.index <= (t1.loc[molecule].max())[0]] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],((t1.max())[0])]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.items():
        for timeIn, timeOut in tOut.items():
            count.loc[timeIn:timeOut]+=1.
    
    return count.loc[molecule[0]:(t1.loc[molecule].max())[0]]


def mpSampleTW(t1,numCoEvents,molecule):
    # Derive average uniqueness over the event's lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].items():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght


def getWeights(d,size):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w


def getWeightsFFD(d, thres):
    # thres>0 drops insignificant weights
    w = [1.]
    k = 1
    while abs(w[-1]) >= thres:  
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    w = np.array(w[ : : -1]).reshape(-1, 1)[1 : ]  
    return w

def fracDiff(series,d,thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights(d,series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]):continue # exclude NAs
            df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df


def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeightsFFD(d,thres)
    width=len(w)-1
    #2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
            df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df