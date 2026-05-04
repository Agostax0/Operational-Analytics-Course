import os.path

import torch.nn as nn
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sympy.abc import alpha
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
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

class EraclitoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EraclitoMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    set_seed()

    PragaDate = 'PragaDate'
    DAILY_TMIN = 'DAILY_TMIN'
    DAILY_TMAX = 'DAILY_TMAX'
    DAILY_PREC = 'DAILY_PREC'
    W_MOND = 'W-MON'
    features_names = [DAILY_TMIN, DAILY_TMAX, DAILY_PREC]

    DAILY_TMIN_OUTLIER = DAILY_TMIN + '_outlier'
    DAILY_TMAX_OUTLIER = DAILY_TMAX + '_outlier'
    DAILY_PREC_OUTLIER = DAILY_PREC + '_outlier'

    DAYS_IN_A_YEAR = 365
    WEEKS_IN_A_YEAR = 52

    dataset = pd.read_csv("combined.csv")

    dataset[PragaDate] = pd.to_datetime(dataset[PragaDate])
    TOTAL_YEARS = int(len(dataset) / DAYS_IN_A_YEAR)
    TOTAL_WEEKS = int(len(dataset) / WEEKS_IN_A_YEAR)

    daily_dataset = dataset

    dataset = (
        dataset
        .set_index(PragaDate)
        .resample(W_MOND)[features_names]
        .mean()
        .reset_index()
    )

    if False:

        dataset_description = dataset.drop(columns=[PragaDate]).describe()

        print(f"dataset description: \n {dataset_description}")

        print("Check for normal distribution of data using the Wilk-Shapiro test")
        for feature in features_names:
            # Check for normal distribution of data using the Wilk-Shapiro test
            _, p = shapiro(dataset[feature].dropna())

            print(f"feature {feature}, p: {p}")
            normally_distributed = "" if p > 0.05 else "Not "
            print(f'\t{normally_distributed}normally distributed ({"" if p<0.05 else normally_distributed}refusing Null Hypothesis)')

        # No feature is normally Distributed
        # feature DAILY_TMIN, p: 2.3353398457236163e-18
        # 	Not normally distributed (refusing Null Hypothesis)
        # feature DAILY_TMAX, p: 2.9995702681637455e-18
        # 	Not normally distributed (refusing Null Hypothesis)
        # feature DAILY_PREC, p: 4.76318265914823e-44
        # 	Not normally distributed (refusing Null Hypothesis)

        # Finding outliers using IQR method, since no feature is normally distributed
        # Outliers must be found within individual months, since temperature values cannot be compared across months
        # July's temperature cannot be compared as an outlier of January
        daily_dataset[DAILY_TMAX_OUTLIER] = daily_dataset.groupby(daily_dataset[PragaDate].dt.month)[DAILY_TMAX].transform(iqr_outliers)
        daily_dataset[DAILY_TMIN_OUTLIER] = daily_dataset.groupby(daily_dataset[PragaDate].dt.month)[DAILY_TMIN].transform(iqr_outliers)
        # daily_dataset[DAILY_PREC_OUTLIER] = daily_dataset.groupby(daily_dataset[PragaDate].dt.month)[DAILY_PREC].transform(iqr_outliers)

        # print(dataset[DAILY_TMAX][dataset[DAILY_TMAX_OUTLIER]])
        # print(dataset[DAILY_TMIN][dataset[DAILY_TMIN_OUTLIER]])
        # print(dataset[DAILY_PREC][dataset[DAILY_PREC_OUTLIER]])

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature C°")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.plot(daily_dataset[PragaDate], daily_dataset[DAILY_TMAX], label=DAILY_TMAX, color='orange', alpha=0.7)
        outliers = daily_dataset[daily_dataset[DAILY_TMAX_OUTLIER]]
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
        ax.plot(daily_dataset[PragaDate], daily_dataset[DAILY_TMIN], label=DAILY_TMIN,color='blue', alpha=0.7)
        outliers = daily_dataset[daily_dataset[DAILY_TMIN_OUTLIER]]
        ax.scatter(outliers[PragaDate], outliers[DAILY_TMIN],
                   color='red', s=10, zorder=5, label='tmin_outliers')
        ax.legend()
        plt.savefig('Tmin_outliers.png')
        print("Saved plot to file: Tmin_outliers.png")

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature C°")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        for feature in [DAILY_TMIN, DAILY_TMAX]:
            ax.plot(daily_dataset[PragaDate], daily_dataset[feature], label=feature, alpha=0.7)

        ax.legend()
        plt.savefig('Tmax-Tmin_figure.png')
        print("Saved plot to file: Tmax-Tmin_figure.png")


    # SARIMAX
    if False:
        split = 0.8
        train_size = int(len(dataset) * split)
        test_size = len(dataset) - train_size
        dataset_indexed = dataset.set_index(PragaDate)
        train, test = dataset_indexed[:train_size], dataset_indexed[train_size:]

        # This method needs stationarity that will be checked with Augumented Dickey Fuller test
        # adf = adfuller(train[DAILY_TMAX].dropna())
        #
        # auto_model = pm.auto_arima(train[DAILY_TMAX],
        #                         test='adf',
        #                         # exogenous=train[[DAILY_TMIN]],
        #                         seasonal=True, m=WEEKS_IN_A_YEAR,
        #                         max_p=3, max_q=3, max_P=2, max_Q=2,
        #                         stepwise=True, trace=True)
        # print(auto_model.order)
        # print(auto_model.seasonal_order)
        # Best model: ARIMA(3,0,1)(2,0,0)[52] intercept
        # Total fit time: 751.500 seconds
        # (3, 0, 1)
        # (2, 0, 0, 52)

        p, d, q = (3,1,1) # auto_model.order
        # There is a strong seasonality correlation so d D = 1
        P, D, Q, s = (1,1,0,WEEKS_IN_A_YEAR) #auto_model.seasonal_order

        if os.path.exists('sarimax.pkl'):
            print("Loaded sarimax model from file")
            sarimax_model_fitted = SARIMAXResults.load('sarimax.pkl')
        else:
            sarimax_model = SARIMAX(train[DAILY_TMAX],
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, s))

            sarimax_model_fitted = sarimax_model.fit(disp=False)
            print("Saved sarimax model to file: sarimax.pkl")
            sarimax_model_fitted.save('sarimax.pkl')

        n_forecast = test_size
        in_sample = sarimax_model_fitted.fittedvalues
        out_of_sample = sarimax_model_fitted.get_forecast(steps=n_forecast).predicted_mean
        full_forecast = pd.concat([in_sample, out_of_sample])

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_xlabel("Week")
        ax.set_ylabel("Temperature C°")
        ax.plot(range(len(dataset_indexed)), dataset_indexed[DAILY_TMAX], label='Actual', alpha=0.7)
        ax.plot(range(len(full_forecast)), full_forecast.values, label='SARIMAX forecast', alpha=0.7)
        ax.axvline(x=len(train), color='red', linestyle='--', label='Train/Test split')

        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Sarimax_figure.png')
        print("Saved plot to file: Sarimax_figure.png")

    # LSTM / MLP
    if False:
        dataset_indexed = dataset.set_index(PragaDate)

        lags = [
            1, 2, 3, 4,
            WEEKS_IN_A_YEAR, WEEKS_IN_A_YEAR*2,
            WEEKS_IN_A_YEAR*3, WEEKS_IN_A_YEAR*4
        ]
        DAILY_TMAX_LAG = 'TMAX_lag_'
        DAILY_TMIN_LAG = 'TMIN_lag_'
        DAILY_PREC_LAG = 'PREC_lag_'

        features = []
        for lag in lags:
            dataset_indexed[f'{DAILY_TMAX_LAG}{lag}'] = dataset_indexed[DAILY_TMAX].shift(lag)
            dataset_indexed[f'{DAILY_TMIN_LAG}{lag}'] = dataset_indexed[DAILY_TMIN].shift(lag)
            dataset_indexed[f'{DAILY_PREC_LAG}{lag}'] = dataset_indexed[DAILY_PREC].shift(lag)

            features.append(f'{DAILY_TMAX_LAG}{lag}')

        target = DAILY_TMAX

        week = dataset_indexed.index.isocalendar().week.astype(float)
        dataset_indexed['sin_week'] = np.sin(2 * np.pi * week / 52)
        dataset_indexed['cos_week'] = np.cos(2 * np.pi * week / 52)

        features.append('sin_week')
        features.append('cos_week')
        dataset_indexed = dataset_indexed.dropna()

        train_split, test_split = 0.85, 0.15

        train_size = int(len(dataset_indexed) * train_split)
        test_size = len(dataset_indexed) - train_size

        train = dataset_indexed[:train_size]
        test = dataset_indexed[train_size:]

        X_scaler = StandardScaler()
        Y_scaler = StandardScaler()

        X_train = X_scaler.fit_transform(train[features])
        Y_train = Y_scaler.fit_transform(train[[target]])

        X_test = X_scaler.transform(test[features])
        Y_test = Y_scaler.transform(test[[target]])

        train_dataset = EraclitoDataset(X_train, Y_train)
        test_dataset = EraclitoDataset(X_test, Y_test)

        # Shuffle = True because the temporal order of features is already encoded in the lags
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = EraclitoMLP(input_dim=len(features))
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        n_epochs = 200

        if os.path.exists('best_mlp.pt'):
            print("Loaded MLP model from file: best_mlp.pt")
            model.load_state_dict(torch.load('best_mlp.pt'))
        else:
            for epoch in range(n_epochs):
                model.train()
                for X_batch, Y_batch in train_loader:
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = loss_fn(pred, Y_batch)
                    loss.backward()
                    optimizer.step()
                    if epoch % 10 == 0:
                        print(f'Finished epoch {epoch}, latest loss {loss}')

            print("Saved MLP model to file: best_mlp.pt")
            torch.save(model.state_dict(), 'best_mlp.pt')

        model.eval()
        with torch.no_grad():
            # In-sample
            train_pred = Y_scaler.inverse_transform(
                model(torch.tensor(X_train, dtype=torch.float32)).numpy()
            )
            # Out-of-sample
            test_pred = Y_scaler.inverse_transform(
                model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            )

        rmse = root_mean_squared_error(test[target].values, test_pred)

        print(f"Final MPL RMSE: {rmse}")

        full_pred = np.concatenate([train_pred, test_pred])
        actual = dataset_indexed[target].values

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.plot(actual, label='Actual', alpha=0.7)
        ax.plot(full_pred, label='MLP forecast', alpha=0.7)
        ax.axvline(x=len(train), color='red', linestyle='--', label='Train/Test split')
        ax.set_xlabel("Week")
        ax.set_ylabel("Temperature C°")
        ax.legend()
        plt.tight_layout()
        plt.savefig('MLP_figure.png')
        print("Saved plot to file: MLP_figure.png")

    # XGBoost / RandomForest
    if True:

        pass

    plt.show(block=True)
