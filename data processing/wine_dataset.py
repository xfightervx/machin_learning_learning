import pandas as pd
import numpy as np
from tools import load_wine_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
def load_data():
    df_wine = load_wine_dataset()
    df_wine.columns =   ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
    X ,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test

def minmaxscalwine():
    X_train, X_test, y_train, y_test = load_data()
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)
    return X_train_norm,X_test_norm,y_train, y_test

def standardscalewine():
    X_train, X_test, y_train, y_test = load_data()
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std,X_test_std,y_train,y_test

