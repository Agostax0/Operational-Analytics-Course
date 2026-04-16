import xgboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBRegressor


def create_dataset(data, seq_len):
    xs, ys = [], []
    for i in range(len(data)- seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

if __name__ == "__main__":

    # rolling window dataset (see MLP)
    df = pd.read_csv('BoxJenkins.csv', usecols=[1], names=['Passengers'], header=0)
    # rolling window dataset (see MLP)
    lookback = 12
    X, y = create_dataset(df["Passengers"].values, lookback)
    nfore = 12
    x_train, _ = X[:-nfore], X[-nfore:]
    y_train, ytest = y[:-nfore], y[-nfore:]
    # fit model
    model = XGBRegressor(objective ='reg:squarederror', n_estimators=1000)
    model.fit(x_train, y_train)

    xinput = x_train[-1]
    yfore = []
    for i in range(12):
        yfore.append(model.predict(xinput.reshape(1, 12))[0])
        xinput = np.roll(xinput, -1)
        xinput[-1] = yfore[-1]
    plt.plot(df.values, label="Actual series")
    plt.plot(range(len(df) - nfore, len(df)), yfore, "-o", label="12-forecast")
    plt.legend()
    plt.show()
    pass