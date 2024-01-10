import os
import pandas as pd

def load_data(data):
    df = pd.read_csv(data)
    df = df.drop('id', axis=1)
    df = df[df.status != 1]
    df.loc[:, 'status'] = df.status / 2.0
    df.loc[:, 'time'] = df.time / 365.0
    df.loc[:, 'trt'] = df.trt - 1
    df.loc[:, 'sex'] = df.sex.map({'f':0.0, 'm':1.0})
    df = df.dropna(axis=0)

    return df