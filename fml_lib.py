import numpy as np
import pandas as pd
import sys
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as mpl
from scipy.stats import rv_continuous, kstest
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
        print(f"Iteration {i}")
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


def getBins(events, close, num_days):
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    #print(px.loc[events_['t1'].values])
    #print(px.loc[events_.index])
    out = pd.DataFrame(index = events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin'] = np.sign(out['ret'])
    vertical_barrier_reached = ((pd.to_datetime(px.loc[events_['t1'].values].index))-pd.to_datetime((px.loc[events_.index].index))) >= pd.Timedelta(days=num_days )
    out['bin'] = out['bin'] * ~vertical_barrier_reached
    #print((px.loc[events_['t1'].values].index))
    print(~vertical_barrier_reached)
    #print(px.loc[events_.index].index)
    return out



def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count
    '''
    #molecule = pd.to_datetime(molecule)
    #t1 = pd.to_datetime(t1)
    #t1 = t1.drop("trgt", axis=1)
    t1 = t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1 = t1.dropna()
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.items():
        count.loc[tIn:tOut]+=1.
    
    return count.loc[molecule[0]:t1[molecule].max()]


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


def getIndMatrix(barIx,t1):
    # Get indicator matrix
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i,(t0,t1) in enumerate(t1.items()):
        for count, (t2, t3) in enumerate(t1.items()):
            indM.loc[t2:t3,count]=1.
    return indM

def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c=indM.sum(axis=1) # concurrency
    u=indM.div(c,axis=0) # uniqueness
    breakpoint()
    avgU=u[u>0].mean() # average uniqueness
    return avgU



def getTrainTimes(t1,testTimes):
    '''
    Given testTimes, find the times of the training observations.
    —t1.index: Time when the observation started.
    —t1.value: Time when the observation ended.
    —testTimes: Times of testing observations.
    '''
    trn=t1.copy(deep=True)
    for i,j in testTimes.iteritems():
        df0=trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test
        df1=trn[(i<=trn)&(trn<=j)].index # train ends within test
        df2=trn[(trn.index<=i)&(j<=trn)].index # train envelops test
        trn=trn.drop(df0.union(df1).union(df2))
    return trn

class PurgedKFold(_BaseKFold):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    '''
    def __init__(self,n_splits=3,t1=None,shuffle=False, pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=shuffle,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo
    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),self.n_splits)]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxT1Idx<X.shape[0]: # right train (with embargo)
                train_indices=np.concatenate((train_indices,indices[maxT1Idx+mbrg:]))
            yield train_indices,test_indices

def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None,pctEmbargo=None, shuffle=False):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,shuffle=shuffle, pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        # think I need to throw in a check for whether or not this is 2 dimensional
        x_train = X.iloc[train]
        #print(len(x_train))
        fit=clf.fit(x_train,y=y.iloc[train],sample_weight=np.squeeze(sample_weight.iloc[train].values))
        x_test = X.iloc[test]
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(x_test)
            score_=-log_loss(y.iloc[test],prob,sample_weight=np.squeeze(sample_weight.iloc[test].values),labels=clf.classes_)
        else:
            pred=fit.predict(x_test)
            score_=accuracy_score(y.iloc[test],pred,sample_weight= np.squeeze(sample_weight.iloc[test].values))
        score.append(score_)
    return np.array(score)


def getTestData(n_features=40,n_informative=10,n_redundant=10,n_samples=10000):
    # generate a random dataset for a classification problem
    from sklearn.datasets import make_classification
    trnsX,cont=make_classification(n_samples=n_samples,n_features=n_features,n_informative=n_informative,n_redundant=n_redundant,random_state=0,shuffle=False)
    df0 = pd.bdate_range(periods = n_samples,end = pd.Timestamp.today())
    trnsX,cont=pd.DataFrame(trnsX,index=df0), pd.Series(cont,index=df0).to_frame('bin')
    df0=['I_'+str(i) for i in range(n_informative)]+['R_'+str(i) for i in range(n_redundant)]
    df0+=['N_'+str(i) for i in range(n_features-len(df0))]
    trnsX.columns=df0
    cont['w']=1./cont.shape[0]
    cont['t1']=pd.Series(cont.index,index=cont.index)
    return trnsX,cont


def get_eVec(dot,varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal,eVec=np.linalg.eigh(dot)
    idx=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[idx],eVec[:,idx]
    #2) only positive eVals
    eVal=pd.Series(eVal,index=['PC_'+str(i+1) for i in range(eVal.shape[0])])
    eVec=pd.DataFrame(eVec,index=dot.index,columns=eVal.index)
    eVec=eVec.loc[:,eVal.index]
    #3) reduce dimension, form PCs
    cumVar=eVal.cumsum()/eVal.sum()
    dim=cumVar.values.searchsorted(varThres)
    eVal,eVec=eVal.iloc[:dim+1],eVec.iloc[:,:dim+1]
    return eVal,eVec


def orthoFeats(dfX,varThres=.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dfZ=dfX.sub(dfX.mean(),axis=1).div(dfX.std(),axis=1) # standardize
    dot=pd.DataFrame(np.dot(dfZ.T,dfZ),index=dfX.columns,columns=dfX.columns)
    eVal,eVec=get_eVec(dot,varThres)
    print(eVal)
    dfP=np.dot(dfZ,eVec)
    return dfP



def featImpMDI(fit,featNames):
    # feat importance based on IS mean impurity reduction
    df0={i:tree.feature_importances_ for i,tree in enumerate(fit.estimators_)}
    #print(df0)
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan) # because max_features=1
    imp=pd.concat({'mean':df0.mean(),'std':df0.std()*df0.shape[0]**-.5},axis=1)
    imp/=imp['mean'].sum()
    return imp
    

def featImpMDA(clf,X,y,cv,sample_weight,t1,pctEmbargo,scoring='neg_log_loss'):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged cv
    scr0,scr1=pd.Series(),pd.DataFrame(columns=X.columns)
    for i,(train,test) in enumerate(cvGen.split(X=X)):
        X0,y0,w0=X.iloc[train,:],y.iloc[train],sample_weight.iloc[train]
        X1,y1,w1=X.iloc[test,:],y.iloc[test],sample_weight.iloc[test]
        fit=clf.fit(X=X0,y=y0,sample_weight=w0.values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1)
            scr0.loc[i]=-log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
        else:
            pred=fit.predict(X1)
            scr0.loc[i]=accuracy_score(y1,pred,sample_weight=w1.values)
        for j in X.columns:
            X1_=X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) # permutation of a single column
            if scoring=='neg_log_loss':
                prob=fit.predict_proba(X1_)
                scr1.loc[i,j]=-log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
            else:
                pred=fit.predict(X1_)
                scr1.loc[i,j]=accuracy_score(y1,pred,sample_weight=w1.values)
    imp=(-scr1).add(scr0,axis=0)
    if scoring=='neg_log_loss':imp=imp/-scr1
    else:imp=imp/(1.-scr1)
    imp=pd.concat({'mean':imp.mean(),'std':imp.std()*imp.shape[0]**-.5},axis=1)
    return imp,scr0.mean()



def auxFeatImpSFI(featNames,clf,trnsX,cont,scoring,cvGen):
    imp=pd.DataFrame(columns=['mean','std'])
    for featName in featNames:
        clf.n_jobs = -1
        df0=cvScore(clf,X=trnsX[[featName]],y=cont['bin'],sample_weight=cont['w'],scoring=scoring,cvGen=cvGen)
        imp.loc[featName,'mean']=df0.mean()
        imp.loc[featName,'std']=df0.std()*df0.shape[0]**-.5
    return imp


def featImportance(trnsX,cont,n_estimators=1000,cv=10,max_samples=1.,pctEmbargo=0,scoring='accuracy',method='SFI',minWLeaf=0.,**kargs):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    #1) prepare classifier,cv. max_features=1, to prevent masking
    clf=DecisionTreeClassifier(criterion='entropy',max_features=1,class_weight='balanced',min_weight_fraction_leaf=minWLeaf)
    clf=BaggingClassifier(estimator=clf,n_estimators=n_estimators,max_features=1.,max_samples=max_samples,oob_score=True)
    fit=clf.fit(X=trnsX,y=cont['bin'],sample_weight=cont['w'].values)
    oob=fit.oob_score_
    if method=='MDI':
        imp=featImpMDI(fit,featNames=trnsX.columns)
        oos=cvScore(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring).mean()
    elif method=='MDA':
        imp,oos=featImpMDA(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring)
    elif method=='SFI':
        cvGen=PurgedKFold(n_splits=cv,t1=cont['t1'],pctEmbargo=pctEmbargo)
        oos=cvScore(clf,X=trnsX,y=cont['bin'],sample_weight=cont['w'],scoring=scoring,
        cvGen=cvGen).mean()
        clf.n_jobs=1 # paralellize auxFeatImpSFI rather than clf
        imp=auxFeatImpSFI(featNames=trnsX.columns, clf=clf,trnsX=trnsX,cont=cont,scoring=scoring,cvGen=cvGen)
    return imp,oob,oos


def plotFeatImportance(pathOut,imp,method,tag=0,simNum=0,**kargs):
    # plot mean imp bars with std
    mpl.figure(figsize=(10,imp.shape[0]/5.))
    imp=imp.sort_values('mean',ascending=True)
    ax=imp['mean'].plot(kind='barh',color='b',alpha=.25,xerr=imp['std'],
    error_kw={'ecolor':'r'})
    if method=='MDI':
        mpl.xlim([0,imp.sum(axis=1).max()])
        mpl.axvline(1./imp.shape[0],linewidth=1,color='r',linestyle='dotted')
        ax.get_yaxis().set_visible(False)
    for i,j in zip(ax.patches,imp.index):ax.text(i.get_width()/2,i.get_y()+i.get_height()/2,j,ha='center',va='center',color='black')
    mpl.title('tag='+str(tag)+' | simNum='+str(simNum))
    mpl.savefig(pathOut+'featImportance_'+str(simNum)+'.png',dpi=100)
    mpl.clf();mpl.close()
    return



def clfHyperFit(feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,0,1.],rndSearchIter=0,n_jobs=-1,pctEmbargo=0,**fit_params):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import BaggingClassifier
    if set(lbl.values)=={0,1}:
        scoring='f1' # f1 for meta-labeling
    else:
        scoring='neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data
    inner_cv=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    if rndSearchIter==0:
        gs=GridSearchCV(estimator=pipe_clf,param_grid=param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs)
    else:
        gs=RandomizedSearchCV(estimator=pipe_clf,param_distributions=param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs,n_iter=rndSearchIter)
    gs=gs.fit(feat,lbl,**fit_params).best_estimator_ # pipeline
    #2) fit validated model on the entirety of the data
    if bagging[1]>0:
        gs=BaggingClassifier(estimator=MyPipeline(gs.steps),n_estimators=int(bagging[0]),max_samples=float(bagging[1]),max_features=float(bagging[2]),n_jobs=n_jobs)
        gs=gs.fit(feat,lbl,sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs=Pipeline([('bag',gs)])
    return gs



class MyPipeline(Pipeline):
    def fit(self,X,y,sample_weight=None,**fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight']=sample_weight
        return super(MyPipeline,self).fit(X,y,**fit_params)
    


class logUniform_gen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self,x):
        return np.log(x/self.a)/np.log(self.b/self.a)

def logUniform(a=1,b=np.exp(1)):
    return logUniform_gen(a=a,b=b,name='logUniform')

def averageActiveSignals(signals):
    tPnts=set(signals['t1'].dropna().values)
    tPnts=tPnts.union(signals.index.values)
    tPnts=list(tPnts);tPnts.sort()
    out = pd.Series()
    for loc in tPnts:
        df0=(signals.index.values<=loc)&((loc<signals['t1'])|pd.isnull(signals['t1']))
        act=signals[df0].index
        if len(act)>0:
            out[loc]=signals.loc[act,'signal'].mean()
        else:
            out[loc]=0
    return out

# Derives HHI concentration, i.e. what percent of the market a firm makes up
def getHHI(betRet):
    if betRet.shape[0]<=2:
        return np.nan
    wght=betRet/betRet.sum()
    hhi=(wght**2).sum()
    hhi=(hhi-betRet.shape[0]**-1)/(1.-betRet.shape[0]**-1)
    return hhi

def computeDD_TuW(series,dollars=False):
    # compute series of drawdowns and the time under water associated with them
    df0=series.to_frame('pnl')
    df0['hwm']=series.expanding().max()
    df1=df0.groupby('hwm').min().reset_index()
    df1.columns=['hwm','min']
    df1.index=df0['hwm'].drop_duplicates(keep='first').index # time of hwm
    df1=df1[df1['hwm']>df1['min']] # hwm followed by a drawdown
    if dollars:
        dd=df1['hwm']-df1['min']
    else:
        dd=1-df1['min']/df1['hwm']
    tuw=((df1.index[1:]-df1.index[:-1])/np.timedelta64(365,'D')).values# in years
    tuw=pd.Series(tuw,index=df1.index[:-1])
    return dd,tuw


def get_co_events(close,events):
    '''
    first = True
    last_index = 0
    overlap_series = []
    for index, row in close.items():
        if first:
            first = False
        else:
            molecule = [last_index, index]
            if molecule[0] in t1.index and molecule[1] in t1.index:
                unique_count = mpNumCoEvents(close, t1, molecule)
                print(unique_count)
                overlap_series.append(unique_count)
            else:
                print("Not in there")
        last_index = index
    overlap_df = pd.concat(overlap_series)
    return overlap_df
    '''
    overlap_df = mpNumCoEvents(close.index,events['t1'], events.index)
    print(overlap_df)
    return overlap_df