import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

if __name__ == "__main__":
    og_traffico = pd.read_csv('traffico16.csv', usecols= ['set2'])
    traffico_meta1 = og_traffico[0:13]
    traffico_meta2 = og_traffico[16::]

    print(og_traffico.shape)
    print(traffico_meta1.shape)
    print(traffico_meta2.shape)

    plt.figure()
    plt.plot(og_traffico, label = 'original', color='red')
    plt.plot(traffico_meta1, label = 'missing', color='blue')
    plt.plot(traffico_meta2, label = 'missing', color='blue')
    plt.legend()
    # Concat vertically (axis=0), keeping original indices
    concatenated_df = pd.concat([traffico_meta1, traffico_meta2], axis=0)

    # Reindex to full range so missing rows (13,14,15) become NaN
    full_index = range(len(og_traffico))
    concatenated_df = concatenated_df.reindex(full_index)

    ds = concatenated_df['set2']

    t1_f = ds.ffill()
    t2_b = ds.bfill()
    t3_m = ds.fillna(ds.mean())
    t4_interpolate = ds.interpolate()

    plt.figure()
    plt.plot(og_traffico['set2'], label='original', color='red', linestyle='--')
    plt.plot(t4_interpolate, label='interpolated', color='blue')
    plt.plot(t1_f, label='ffill', color='green')
    plt.plot(t2_b, label='bfill', color='orange')
    plt.legend()



    df = og_traffico.fillna(value=og_traffico.mean())
    result = seasonal_decompose(df, model='additive', period=7)
    observed = result.observed
    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    std = resid.std()

    plt.figure()
    plt.plot(resid, "o", label='datapoints')
    plt.hlines(0,0,len(resid))
    plt.hlines(1.5*std, 0, len(resid), color='red', label='std limits')
    plt.hlines(-1.5*std, 0, len(resid), color='red')
    plt.legend()
    plt.show()

    pass