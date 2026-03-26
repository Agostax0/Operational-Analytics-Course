# Diebold mariano dice se due modelli sono statisticamente equivalenti
# Holt Winters e Theta => fare previsioni => applicare diebold mariano

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from dm_test import dm_test

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) # MAPE
    me = np.mean(forecast - actual) # ME
    mae = np.mean(np.abs(forecast - actual)) # MAE
    mpe = np.mean((forecast - actual)/actual) # MPE
    rmse = np.mean((forecast - actual)**2)**.5 # RMSE
    corr = np.corrcoef(forecast, actual)[0,1] # correlation coeff
    acf1 = acf(forecast-actual)[1] # ACF1
    return {'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 'corr':corr}

if __name__ == "__main__":
    df = pd.read_csv('M3C_monthly.csv')
    rawdata = df.iloc[490, 6:].values.astype(float)
    train, test = rawdata[:-12], rawdata[-12:]
    seasonal_periods = 12


    model_holt_winters = ExponentialSmoothing(train, seasonal_periods=seasonal_periods, trend="add",
                                seasonal="mul",
                                damped_trend=True,
                                use_boxcox=True,
                                initialization_method="estimated")

    hw_fit = model_holt_winters.fit()
    hw_fore = hw_fit.forecast(steps = len(test))


    model_theta = ThetaModel(train, period=seasonal_periods)
    theta_fit = model_theta.fit()
    theta_fore = theta_fit.forecast(steps=len(test))

    plt.figure()
    plt.plot(hw_fore, label = 'hw')
    plt.plot(np.arange(0, len(test)),theta_fore, label = 'theta')
    plt.plot(test, label = 'actual')
    plt.legend()


    hw_rmse = forecast_accuracy(hw_fore, test)['rmse']
    theta_rmse = forecast_accuracy(theta_fore, test)['rmse']

    rt = dm_test(test, hw_fore, theta_fore, crit='MSE')
    print(rt)

    print(f"Rejecting Null Hypothesys {rt[0] < -1.96 or rt[0] > 1.96}")

    plt.show()
