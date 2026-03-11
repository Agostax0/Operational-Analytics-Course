import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def difference(data, interval):
    return [data[i] - data[i-interval] for i in range(interval,len(data))]
# invert difference
def invert_difference(orig_data, diff_data, interval):
    inverted = [diff_data[i]+orig_data[i] for i in range(interval)]
    for i in range(interval,len(diff_data)):
        inverted = np.append(inverted,diff_data[i]+inverted[i-interval])
    return inverted

if __name__ == "__main__":
    airline = pd.read_csv("./BoxJenkins.csv", usecols=["Passengers"]) # Carica solo questa colonna

    training_set = np.array(airline)
    training_set = training_set[0:len(training_set)-12]

    test_set = np.array(airline)
    test_set = test_set[len(test_set) - 12::]

    print(f"dt: {airline.shape}")
    print(f"train: {training_set.shape}")
    print(f"test: {test_set.shape}")

    training_log_set = [np.log(x) for x in training_set]
    training_diff1_log_set = difference(training_log_set, 1)
    training_diffm_diff1_log_set = difference(training_diff1_log_set, 12)

    plt.figure('training and test')
    plt.plot(training_set, color='blue', label = 'training')
    plt.plot(np.arange(132,144), test_set, color='red', label = 'test')
    plt.legend()

    plt.figure('log, diff1, diffm')
    plt.plot(training_log_set, color='red', label = 'log')
    plt.plot(training_diff1_log_set, color='green', label = 'diff1_log')
    plt.plot(training_diffm_diff1_log_set, color='blue', label = 'diffm_diff1_log')
    plt.legend()


    print(np.average(training_diffm_diff1_log_set))

    predicted_diffm_diff1_log_set = np.append(training_diffm_diff1_log_set, np.zeros(12))

    print(f"prediction: {predicted_diffm_diff1_log_set.shape}")

    predicted_diff1_log_set = invert_difference(training_diff1_log_set, predicted_diffm_diff1_log_set, 12)

    predicted_log_set = invert_difference(training_log_set, predicted_diff1_log_set, 1)

    predicted_set = [np.exp(x) for x in predicted_log_set]

    plt.figure('inv_log, inv_diff1, inv_diffm')
    plt.plot(predicted_set, color='red', label='inv_log')
    plt.plot(predicted_log_set, color='green', label='inv_diff1')
    plt.plot(predicted_diff1_log_set, color='blue', label='inv_diffm')
    plt.legend()

    plt.figure('actual, predicted')
    display_predicted = predicted_set[len(predicted_set) - 12::]
    plt.plot(display_predicted, color='red', label='predicted')
    plt.plot(test_set, color='green', label='actual')

    plt.show()

