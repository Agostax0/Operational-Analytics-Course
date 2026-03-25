import pandas as pd, numpy as np
import matplotlib.pyplot as plt # AR(2) series generation
np.random.seed(995)
n = 200 # for stationarity: phi2 + phi1 < 1, phi2 - phi1 < 1, |phi2| < 1
phi = [0.6, -0.3]
# Generate series
errors = np.random.normal(0, 1, n)
y = np.zeros(n)
for t in range(2, n):

    reg_sum = 0
    for phi_index in range(len(phi)):
        reg_sum = reg_sum + phi[phi_index] * y[t - phi_index]

    y[t] = reg_sum + errors[t]

plt.plot(y)
plt.title("AR(2) series")
plt.xlabel("time")
plt.ylabel("values")
plt.show()

