#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from sklearn.preprocessing import StandardScaler


def getSession():
    cloud_config = {
        'secure_connect_bundle': 'secure-connect-storesales (1).zip'
    }
    auth_provider = PlainTextAuthProvider('egiycAaYmQSrxipFZzEiJmog',
                                          'b6iKLJrzf4l,rewyA4Wf,JYSvv,D73ACcPSseZadGuq+X1dZ1AXX,vrUioXWOk47sQ2bx_YANYz,Z0E2Cv_oRGM5fjgw4d6ReL-hFnu2ewRqNzXCuXYPUm+IqQ5Mp8Ug')
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect('SalesPrediction')
    return session

def verifyUser(name,email,session):
    row=data=session.execute("select * from modelusers")
    flag=True
    for i in row:
        if name==i.name or email==i.email:
            flag=False
            break
    return flag

def authenticateUser(name,password,session):
    data=session.execute("select * from modelusers")
    for row in data:
        if row.name==name and row.password==password:
            return True
    else:
        return False
def saveUser(name, email, password, session):
    session.execute("insert into modelusers(name,email,password) values(%s,%s,%s);", (name, email, password))


def seperateFeatures(df):
    data1 = df.select_dtypes(include=['int64','float64'])
    data2 = df.select_dtypes(exclude=['float64','int64'])
    data1[data2.columns] = data2
    return data1


def DFboxplot(df):
    n=0
    for i in set(df.dtypes.values):
        if i=="int64" or i=="float64":
            n=n+df.dtypes.value_counts()[i]
    df=seperateFeatures(df)
    fig = plt.figure(figsize=(20, 20))
    pn = 1
    for col in df:
        if pn <= n:
            plt.subplot(3, (np.round(n / 3)+1), pn)
            g = sns.boxplot(df[col])
            plt.xlabel(col, fontsize=15)
        pn += 1
    plt.show()


def DFdistplot(df):
    n=0
    for i in set(df.dtypes.values):
        if i=="int64" or i=="float64":
            n=n+df.dtypes.value_counts()[i]
    df=seperateFeatures(df)
    fig = plt.figure(figsize=(20, 20))
    pn = 1
    for col in df:
        if pn <= n:
            plt.subplot(3, (np.round(n / 3)+1), pn)
            g = sns.distplot(df[col])
            plt.xlabel(col, fontsize=15)
        pn += 1
    plt.show()


def removeOutier(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_cap = Q1 - 1.5 * IQR
        higher_cap = Q3 + 1.5 * IQR
        df.loc[(df[col] > higher_cap), col] = higher_cap
        df.loc[(df[col] < lower_cap), col] = lower_cap
    return df


def myFeatures(X, model, n):
    features = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n:]
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

