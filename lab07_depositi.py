import numpy as np
from Cython.Shadow import typeof
from pandas import DataFrame

DISTANZE_DEPOSITI_INDEX=20

DISTANZE_CLIENTI_DEPOSITO_INDEX=21

# tra le colonne della matrice
# guardare quanto fa la somma => somma totale
# somma totale * 1.3 / 25

def find_min_index_in_row(dist: DataFrame, visited, i):
    min_cost = 999999
    min_index = -1
    for row in range(len(dist.iloc[i,:].values)):
        x = 0
        if dist.iloc[i,:].values[row] < min_cost and not visited[row]:
            min_cost = dist.iloc[i,:].values[row]
            min_index = row

    return min_index

def initial_solution(dist: DataFrame, demands, ntrucks, cap):
    cost = 0
    visited = np.zeros(len(demands), dtype=bool)
    visited[20] = True  # depot
    routes = [[20] for _ in range(ntrucks)]

    for k in range(ntrucks):
        i = 20
        load = 0

        while True:
            j = find_min_index_in_row(dist, visited, i)

            if j == -1 or (load + demands[j]  > cap):
                break

            visited[j] = True
            routes[k].append(j)
            load += demands[j]           # Bug 1 fixed: scalar, not demands[j][0]
            cost += dist.iloc[i, j]      # Bug 2 fixed: accumulate edge cost directly
            i = j

        # Add return-to-depot cost
        cost += dist.iloc[i, 20]
        routes[k].append(20)

    return routes, cost
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    df = pd.read_csv('mat52.csv')
    scatter = pd.read_csv('customers_scatter.csv')

    distanze_depositi_df = df.iloc[:DISTANZE_DEPOSITI_INDEX + 1 , :DISTANZE_DEPOSITI_INDEX + 1]

    predizioni = pd.read_csv('predizioni.csv')
    richieste_array = []
    sum = 0
    for i in range(DISTANZE_DEPOSITI_INDEX + 1):
        sum += (predizioni[str(i)] / 5)
        num = 0
        num += predizioni[str(i)][0]
        richieste_array.append(num)


    sum = np.ceil(sum)

    plt.figure()
    plt.scatter(x=scatter['x'], y=scatter['y'], )

    print(typeof(richieste_array[0]))

    sol, _ = initial_solution(distanze_depositi_df, richieste_array, 10, 50)

    print(sol)

    plt.show(block=True)


