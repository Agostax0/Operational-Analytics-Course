import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro

def average(df):
    return np.mean(df)

def std(df):
    return np.std(df)

def median(df):
    return np.median(df)

def IQR(df):
    quantiles = df.quantile([0.25, 0.5, 0.75])
    return quantiles[0.75] - quantiles[0.25]

def outlier1(name, data, avg, std):
    print(f"{name} - {name}_avg > 1.5 * {name}_std")
    return np.array(data[np.abs(data - avg) > 1.5 * std])

def outlier2(name, data, median, iqr):
    print(f"{name} - {name}_median > 1.5 * {name}_iqr")
    return np.array(data[np.abs(data - median) > 1.5 * iqr])

def null_hypothesis(name, df, avg, std, median, iqr):
    stat, p = shapiro(df)
    if(p > 0.05):
        print(f'{name} Normal (not refusing Null Hypothesis)')
        # Usare std per calcolare outlier
        print('std-calculated outliers')
        print(outlier1(name, df, avg, std))

    else:
        print(f'{name} Non normal (refusing Null Hypothesis)')
        # Usare iqr per calculare outlier
        print('iqr-calculated outliers')
        print(outlier2(name, df, median, iqr))


if __name__ == "__main__":
    df = pd.read_csv("./traffico16.csv")
    month1 = 'set1'
    month2 = 'set2'

    df_month1 = df[month1].dropna()
    print(f"-----{month1}-----")
    month1_average = average(df_month1)
    print(f"avg: {month1_average}")
    month1_std = std(df_month1)
    print(f"std: {month1_std}")
    month1_median = median(df_month1)
    print(f"median: {month1_median}")
    month1_iqr = IQR(df_month1)
    print(f"iqr: {month1_iqr}")

    print(f"-----{month2}-----")
    df_month2 = df[month2].dropna()
    month2_average = average(df_month2)
    print(f"avg: {month2_average}")
    month2_std = std(df_month2)
    print(f"std: {month2_std}")
    month2_median = median(df_month2)
    print(f"median: {month2_median}")
    month2_iqr = IQR(df_month2)
    print(f"iqr: {month2_iqr}")

    print("-----Outliers-----")
    month1_es1 = outlier1(month1, df_month1, month1_average, month1_std)
    print(month1_es1)
    month1_es2 = outlier2(month1, df_month1, month1_median, month1_iqr)
    print(month1_es2)
    print("")
    month2_es1 = outlier1(month2, df_month2, month2_average, month2_std)
    print(month2_es1)
    month2_es2 = outlier2(month2, df_month2, month2_median, month2_iqr)
    print(month2_es2)

    plt.figure()
    plt.title(f"Boxplot {month1} e {month2}")
    df[[month1,month2]].boxplot()


    print("-----Normal Distribution and outliers-----")

    #qqplot(df_month1.sort_values(), line='q')
    #qqplot(df_month2.sort_values(), line='q')

    null_hypothesis(month1, df_month1, month1_average, month1_std, month1_median, month1_iqr)
    print("")
    null_hypothesis(month2, df_month2, month2_average, month2_std, month2_median, month2_iqr)

    plt.show()