import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

if __name__ == "__main__":
    data = pd.read_csv("BoxJenkins.csv", usecols=["Passengers"]).values.flatten()

    p = adfuller(data)

    print(f"{p[1] > 0.05}") # Null hypothesis


    data = pd.read_csv("BoxJenkins.csv", usecols=["Passengers"]).values.flatten()
    data_log = np.log(data)
    data_diff1_log = np.diff(data_log)  # or [ylog[i]-ylog[i-1] for i in range(1,len(ylog))]
    data_diffm_diff1_log = [data_diff1_log[t] - data_diff1_log[t - 12] for t in range(12, len(data_diff1_log))]


    p = adfuller(data_diffm_diff1_log) # senza trend e stagionalita
    print(f"{p[1] > 0.05}") # Null hypothesis


    model = ARIMA(data_diffm_diff1_log[:-12], order=(2,1,0))
    model_fit = model.fit()

    #print(model_fit.summary())
    pred_diffm_diff1_log = model_fit.predict(0, len(data_diffm_diff1_log) - 1)
    pred_diffm_diff1_log[:2] = data_diffm_diff1_log[:2]

    res = []

    for i in range(12):
        res.append(data_diff1_log[i])
    for i in range(len(pred_diffm_diff1_log)):
        res.append(pred_diffm_diff1_log[i])

    pred_diff1_log = res

    for i in range(12, len(pred_diff1_log)):
        pred_diff1_log[i] += pred_diff1_log[i - 12]
    # Invert diff(1)
    pred_log = [data_log[0]] + pred_diff1_log
    for i in range(1, len(pred_log)):
        pred_log[i] += pred_log[i - 1]

    pred = np.exp(pred_log)


    plt.plot(data[:132], label="Actual")
    plt.plot(np.arange(132,144), data[132:144], label="Actual")
    plt.plot(np.arange(132,144),pred[132:144], label="Prediction")
    plt.legend()
    plt.show()