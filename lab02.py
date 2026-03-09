import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.seasonal import seasonal_decompose

def diff(data, interval):
    res = []
    for i in range(len(data)):
        if i - interval < 0:
            continue
        else:
            res.append(data[i] - data[i-interval])
    return res

if __name__ == "__main__":
    box_jenkins = pd.read_csv("./BoxJenkins.csv", usecols=["Passengers"]) # Carica solo questa colonna
    fil_rouge = pd.read_csv("./FilRouge.csv")
    jewerly = pd.read_csv("./jewelry.csv")
    #plt.rcParams['figure.figsize'] = (10.0, 6.0)
    #ds = box_jenkins[box_jenkins.columns[0]] # Converte in una data series
    #result = seasonal_decompose(ds, model='multiplicative', period=12) # La seasonal_decompose ha bisogno di una data series
    # Multiplicative =>
    # period => gli do una stagionalità da guardare
    #Trend è la media mobile
    #Seasonal è il pattern che si adatta meglio alla serie
    #Residual identifica della randomicità
    #result.plot()
    # print(box_jenkins[])
    # plt.plot(box_jenkins['Passengers'])
    # plt.plot(fil_rouge['sales'])
    # plt.plot(jewerly['number'])
    #plt.show()

    #ds = pd.read_csv('BoxJenkins.csv', header=0)
    #X = ds.Passengers.values
    #train_size = int(len(X) * 0.66)
    #train, test = X[0:train_size], X[train_size:len(X)]
    #print('Observations: %d' % (len(X)))
    #print('Training Observations: %d' % (len(train)))
    #print('Testing Observations: %d' % (len(test)))
    #plt.plot(train)
    #plt.plot([None for i in train] + [x for x in test])
    #plt.show()

    diff_box_jenkins = diff(box_jenkins.values, 1)
    # Diff 1 toglie i trend lineri
    log_box_jenkins = [np.log(x) for x in box_jenkins.values]
    log_diff_box_jenkins = diff(log_box_jenkins, 1)
    diff_m_box_jenkins = diff(box_jenkins.values, 12)

    plt.plot(np.array(diff_box_jenkins))
    #plt.plot(np.array(log_box_jenkins))
    plt.plot(np.array(log_diff_box_jenkins))
    #plt.plot(np.array(diff_m_box_jenkins))
    plt.plot(box_jenkins)
    plt.show()
