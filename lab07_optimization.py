import matplotlib.pyplot
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def replace_covid_months(group: pd.DataFrame) -> pd.DataFrame:

    COVID_MONTHS = [
        'Feb-20',
        'Mar-20',
        'Apr-20',
    ]

    REPLACING_MONTHS = [
        'Feb-',
        'Mar-',
        'Apr-',
    ]

    for rm, m in zip(REPLACING_MONTHS, COVID_MONTHS):
        mask_reference = group['month'].str.contains(rm) & (group['month'] != m)
        mask_covid     = (group['month'] == m)

        if mask_reference.any() and mask_covid.any():
            mean_val = round(group.loc[mask_reference, 'val'].mean(), 4)
            group.loc[mask_covid, 'val'] = mean_val

    return group


if __name__ == '__main__':

    ID_SERIE = 'idserie'
    df = pd.read_csv('serie_covid_new.csv')
    df['val'] = df['val'].astype(float)

    series_array: list[pd.DataFrame] = [
        group.reset_index(drop=True)
        for _, group in df.groupby(ID_SERIE)
    ]

    series_array = [replace_covid_months(s) for s in series_array]


    p, d, q = (1, 1, 1)
    P, D, Q, s = (1, 1, 1, 12)


    val_size = len(series_array[0].values)
    test_size = 3

    train = series_array[0].iloc[:val_size-test_size]['val']

    exog = []
    for i in range(len(series_array)):
        exog.append(series_array[i].iloc[:val_size-test_size]['val'])


    sarimax_model = SARIMAX(train,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s))

    sarimax_model_fitted = sarimax_model.fit(disp=False)

    n_forecast = test_size
    in_sample = sarimax_model_fitted.fittedvalues
    out_of_sample = sarimax_model_fitted.get_forecast(steps=n_forecast).predicted_mean
    full_forecast = pd.concat([in_sample, out_of_sample])



    for i in range(len(series_array)-1):
        matplotlib.pyplot.plot(series_array[i].iloc[:val_size-test_size]['val'])


    matplotlib.pyplot.plot(series_array[len(series_array)-1]['val'], label='test')

    matplotlib.pyplot.plot(full_forecast, linestyle='--', label='Forecast')

    matplotlib.pyplot.legend()

    matplotlib.pyplot.show(block=True)