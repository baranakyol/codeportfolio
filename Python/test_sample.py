
# coding: utf-8

# In[2]:

import pytest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import numpy as np

def builtindivide(x):
    return float(x)/float(8)

def test_builtin():
    assert builtindivide(2) == float(0.25)

def test_numpy():
    assert np.array(2.)/np.array(8.) == 0.25

def test_numpy():
    assert numpydivide(2) == 0.25

    
def test_txt():
    assert len(open("input.txt").read()) == 19







def test_sklearn():
    data = load_iris()
    X, y = data.data, data.target
    neigh = KNeighborsClassifier()
    neigh.fit(X,y)
    assert np.mean(cross_val_score(neigh,X,y,cv=5)) >= 0.70



