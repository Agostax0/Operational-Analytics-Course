# random forest for making predictions for regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error

import xgboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
def create_dataset(data, seq_len):
    xs, ys = [], []
    for i in range(len(data)- seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)


if __name__ == "__main__":

    # rolling window dataset (see MLP)
    df = pd.read_csv('M3C_monthly.csv')
    rawdata_x = np.arange(len(df.iloc[505, 6:].values))
    rawdata_y = df.iloc[505, 6:].values.astype(float)
    rawdata = pd.DataFrame(rawdata_y.transpose(), rawdata_x.transpose())
    df = rawdata




    # rolling window dataset (see MLP)
    lookback = 12
    X,y = create_dataset(rawdata_y.transpose(),lookback)
    nfore = 12
    x_train, _ = X[:-nfore], X[-nfore:]
    y_train, ytest = y[:-nfore], y[-nfore:]
    # define the model
    RFmodel = RandomForestRegressor(n_estimators=500, random_state=1)
    RFmodel.fit(x_train, y_train)

    xinput = x_train[-1]
    yfore = []
    for i in range(12):
        yfore.append(RFmodel.predict(xinput.reshape(1,12))[0])
        xinput = np.roll(xinput,-1)
        xinput[-1] = yfore[-1]
    mse = mean_absolute_error(ytest, yfore)
    print("MSE={}".format(mse))
    plt.plot(df.values,label="Actual series")
    plt.plot(range(len(df)-nfore,len(df)),yfore,"-o",label="12-forecast")
    plt.legend()


    if True:
        n_nodes = []
        max_depths = []
        for ind_tree in RFmodel.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)
        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')
        # plot first tree (index 0)
        from sklearn.tree import plot_tree

        fig = plt.figure(figsize=(15, 10))
        plot_tree(RFmodel.estimators_[0],
                  max_depth=2,
                  feature_names=None,#rawdata.columns[:-1],
                  class_names=False,#dataset.columns[-1],
                  filled=True, impurity=True,
                  rounded=True)


    # find the columns with most predictive power
    rfe = RFE(RFmodel, n_features_to_select=4)  # Recursive Feature
    fit = rfe.fit(X, y)
    names = [f"t-{i+1}" for i in range(lookback)]
    predictors = []
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            predictors.append(names[i])
    print("Columns with predictive power:", predictors)

    plt.show()