import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    box_jenkins = pd.read_csv("./BoxJenkins.csv")
    fil_rouge = pd.read_csv("./FilRouge.csv")
    jewerly = pd.read_csv("./jewelry.csv")

    # print(box_jenkins[])

    plt.plot(box_jenkins['Passengers'])
    plt.plot(fil_rouge['sales'])
    plt.plot(jewerly['number'])

    plt.show()
