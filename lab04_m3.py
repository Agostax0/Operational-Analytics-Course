import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

if __name__ == "__main__":

    # Load the dataset
    df = pd.read_csv("M3C_monthly.csv")
    rawdata = df.iloc[490,6:].values.astype(float)
    train,test = rawdata[:-12], rawdata[-12:]
    logdata = np.log(train) # log transform
    logdiff = pd.Series(logdata).diff() # logdiff transform
    # forecast, you can do better

    model = ARIMA(logdiff, order=(1,1,1))
    model_fit = model.fit()
    print(model_fit.summary())
    fore = model_fit.forecast(len(test))
    print(len(fore))
    # fore = pd.Series(0, index=range(132, 132 + 12))
    # Postprocessing, reconstruction (here very pythonic, otherwise plain loops)
    logdiff[0] = logdata[0] # set first series entry
    reclogdata = pd.concat([logdiff,fore]).cumsum()
    recdata = np.exp(reclogdata)

    print(recdata)

    plt.figure()
    plt.plot(recdata)

    plt.figure()
    plt.plot(rawdata[:-12], label
    ="train",linewidth=5)
    plt.plot(recdata[:-12],label
    ="prediction")
    plt.plot(range(len(rawdata)-12,len(rawdata)),recdata[-12:],label
    ="forecast")
    plt.plot(range(len(rawdata)-12,len(rawdata)),rawdata[-12:], label
    ="test")
    plt.title("M3 series"),plt.xlabel("time"),plt.ylabel("value"),plt.ylim(0,10000)
    plt.legend()
    plt.show()