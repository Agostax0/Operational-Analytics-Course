import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram

if __name__ == "__main__":

    y = pd.read_csv('jewelry.csv')
    y = y.iloc[:,1].astype(float)

    fs = 1.0

    f, Pxx = periodogram(y, fs=fs)
    plt.semilogy(f, Pxx)
    plt.ylim([100,100_000_000])
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Airline passengers periodogram')
    plt.grid(True)

    peak_freq = f[np.argmax(Pxx)]
    print(f"Dominant freq: {peak_freq}, cycles: {1/peak_freq}")

    plt.show()

    pass