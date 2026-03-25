import numpy as np
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv('M3C_monthly.csv')
    rawdata = df.iloc[490,6:].values.astype(float)
    train,test = rawdata[:-12], rawdata[-12:]

    model = pm.auto_arima(train, start_p=1, start_q=1, test='adf',
                          max_p=3, max_q=3, m=4, start_P=0, seasonal=True,
                          d = None, D=1, trace=True, error_action='ignore',
                          suppress_warnings=True, stepwise=True
    )
    print(model.summary())

    morder = model.order
    mseasorder = model.seasonal_order

    fitted = model.fit(train)
    n_periods = 12
    yfore, confint = fitted.predict(n_periods, return_conf_int = True)
    ypred = fitted.predict_in_sample()


    plt.plot(rawdata, label = 'data')
    plt.plot(ypred, label = 'pred')
    plt.plot([None for i in ypred] + [x for x in yfore], label = 'fore' )

    plt.figure()
    fitted_series = pd.Series(fitted)
    lower_series = pd.Series(confint[:,0])
    upper_series = pd.Series(confint[:,1])
    plt.plot(yfore)
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha = 0.15)

    plt.legend()
    plt.show()