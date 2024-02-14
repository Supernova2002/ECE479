import numpy as np
import pandas as pd
import sys
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier



def getIndMatrix(barIx,t1):
    # Get indicator matrix
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i,(t0,t1) in enumerate(t1.iteritems()):indM.loc[t0:t1,i]=1.
    return indM



def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c=indM.sum(axis=1) # concurrency
    u=indM.div(c,axis=0) # uniqueness
    avgU=u[u>0].mean() # average uniqueness
    return avgU


# Q: Why is bagging based on random sampling with replacement? Would bagging still reduce a forecast’s variance if sampling were without replacement?
# A : 


# Exercise 3: Build an ensemble of estimators, where the base estimator is a decision tree
if __name__ == "__main__":
    # It's that simple
    ensemble = BaggingClassifier()

    # would derive avgU by taking time index for bars as barIx, a series t1, passing those into getIndMatrix, then passing that output into getAvgUniqueness

    # Produce a bagging classifier that functions like a RF
    clf=DecisionTreeClassifier(criterion='entropy',max_features='auto',class_weight='balanced')
    bc=BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=avgU,max_features=1.)