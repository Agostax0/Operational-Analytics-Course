import torch.nn as nn
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy import stats
from sympy.abc import alpha
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kstest, probplot, shapiro
import matplotlib.dates as mdates


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # if using GPU
    torch.backends.cudnn.deterministic = True  # deterministic ops
    torch.backends.cudnn.benchmark = False     # disable auto-tuner

def iqr_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

if __name__ == "__main__":
    set_seed()

    PragaDate = 'PragaDate'
    DAILY_TMIN = 'DAILY_TMIN'
    DAILY_TMAX = 'DAILY_TMAX'
    DAILY_PREC = 'DAILY_PREC'
    features = [DAILY_TMIN, DAILY_TMAX, DAILY_PREC]

    DAILY_TMIN_OUTLIER = DAILY_TMIN + '_outlier'
    DAILY_TMAX_OUTLIER = DAILY_TMAX + '_outlier'
    DAILY_PREC_OUTLIER = DAILY_PREC + '_outlier'

    DAYS_IN_A_YEAR = 365

    dataset = pd.read_csv("combined.csv")

    dataset[PragaDate] = pd.to_datetime(dataset[PragaDate])
    TOTAL_YEARS = int(len(dataset) / DAYS_IN_A_YEAR)


    dataset_description = dataset.drop(columns=[PragaDate]).describe()

    print(f"dataset description: \n {dataset_description}")

    print("Check for normal distribution of data using the Wilk-Shapiro test")
    for feature in features:
        # Check for normal distribution of data using the Wilk-Shapiro test
        # Shapiro fails for datasets greater than 5000 entries
        _, p = shapiro(dataset[feature].dropna().values[:5000])

        print(f"feature {feature}, p: {p}")
        normally_distributed = "" if p > 0.05 else "Not "
        print(f'\t{normally_distributed}normally distributed ({normally_distributed}refusing Null Hypothesis)')

    # No feature is normally Distributed
    # feature DAILY_TMIN, p: 1.923672196823777e-30
    # 	Not normally distributed (Not refusing Null Hypothesis)
    # feature DAILY_TMAX, p: 5.3386025074974396e-31
    # 	Not normally distributed (Not refusing Null Hypothesis)
    # feature DAILY_PREC, p: 2.644783558246927e-85
    # 	Not normally distributed (Not refusing Null Hypothesis)

    # Finding outliers using IQR method, since no feature is normally distributed
    # Outliers must be found within individual months, since temperature values cannot be compared across months
    # July's temperature cannot be compared as an outlier of January
    dataset[DAILY_TMAX_OUTLIER] = dataset.groupby(dataset[PragaDate].dt.month)[DAILY_TMAX].transform(iqr_outliers)
    dataset[DAILY_TMIN_OUTLIER] = dataset.groupby(dataset[PragaDate].dt.month)[DAILY_TMIN].transform(iqr_outliers)
    # dataset[DAILY_PREC_OUTLIER] = dataset.groupby(dataset[PragaDate].dt.month)[DAILY_PREC].transform(iqr_outliers)

    # print(dataset[DAILY_TMAX][dataset[DAILY_TMAX_OUTLIER]])
    # print(dataset[DAILY_TMIN][dataset[DAILY_TMIN_OUTLIER]])
    # print(dataset[DAILY_PREC][dataset[DAILY_PREC_OUTLIER]])

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature C°")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.plot(dataset[PragaDate], dataset[DAILY_TMAX], label=DAILY_TMAX, color='orange', alpha=0.7)
    outliers = dataset[dataset[DAILY_TMAX_OUTLIER]]
    ax.scatter(outliers[PragaDate], outliers[DAILY_TMAX],
               color='red', s=10, zorder=5, label='tmax_outliers')
    ax.legend()
    plt.savefig('Tmax_outliers.png')
    print("Saved plot to file: Tmax_outliers.png")

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature C°")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.plot(dataset[PragaDate], dataset[DAILY_TMIN], label=DAILY_TMIN,color='blue', alpha=0.7)
    outliers = dataset[dataset[DAILY_TMIN_OUTLIER]]
    ax.scatter(outliers[PragaDate], outliers[DAILY_TMIN],
               color='red', s=10, zorder=5, label='tmin_outliers')
    ax.legend()
    plt.savefig('Tmin_outliers.png')
    print("Saved plot to file: Tmin_outliers.png")

    # SARIMAX

    # LSTM / MLP + Boost / Forest

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature C°")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    for feature in [DAILY_TMIN, DAILY_TMAX]:
        ax.plot(dataset[PragaDate], dataset[feature], label=feature, alpha = 0.7)

    ax.legend()
    plt.savefig('Tmax-Tmin_figure.png')
    print("Saved plot to file: Tmax-Tmin_figure.png")
    plt.show(block = True)
