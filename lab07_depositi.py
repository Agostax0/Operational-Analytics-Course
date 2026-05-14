import numpy as np
from Cython.Shadow import typeof
from pandas import DataFrame

from goto import goto


# >>> DA MIGLIORARE SCAMBI FRA ELEMENTI DI UNA STESSA ROUTE <<<<<<<<<<<<<<
def delta_cost(routes, k, k1, i, j, d):
    deltak = 0
    deltak1 = 0
    if (k != k1):
        # inserisco i prima di j in k1
        deltak1 -= d[routes[k1][j - 1], routes[k1][j]]
        deltak1 += d[routes[k1][j - 1], routes[k][i]] + d[routes[k][i], routes[k1][j]]
        # tolgo i da k
        deltak -= d[routes[k][i - 1], routes[k][i]]
        deltak -= d[routes[k][i], routes[k][i + 1]]
        deltak += d[routes[k][i - 1], routes[k][i + 1]]
    elif (k == k1 and (j - i) != 1):
        deltak -= (d[routes[k][i - 1], routes[k][i]] + d[routes[k][i], routes[k][i + 1]] +
                   d[routes[k][j - 1], routes[k][j]])
        deltak += d[routes[k][i - 1], routes[k][i + 1]] + d[routes[k][j - 1], routes[k][i]] + d[
            routes[k][i], routes[k][j]]

    delta = deltak1 + deltak
    return delta, deltak, deltak1


# prende il nodo dalla pos i della route k e lo mette prima della pos j della route k1
def delta_swap(routes, k, k1, i, j):
    node = routes[k].pop(i)
    if (k != k1 or i > j):
        routes[k1].insert(j, node)
    else:
        routes[k1].insert(j - 1, node)  # ne avevo tolto uno prima
    return


@goto
def local_search(routes, requests, cap, d):
    loads = np.zeros(len(routes))
    rcosts = np.zeros(len(routes))
    # ricostruisce carichi e costi della soluzione iniziale
    for k in range(len(routes)):
        for i in range(1, len(routes[k])):
            loads[k] += requests[routes[k][i]]
            rcosts[k] += d[routes[k][i - 1]][routes[k][i]]

    cont = 0
    label.repeat
    for k in range(len(routes)):
        for i in range(1, len(routes[k]) - 1):  # elemento da spostare
            node = routes[k][i]
            for k1 in range(len(routes)):
                for j in range(1, len(routes[k1])):  # riposiziono prima di j in k1
                    if (k != k1 or abs(i - j) > 0):  # se route diverse o nodi diversi stessa route
                        if (k == k1 or loads[k1] + requests[node] <= cap):  # TODO: spostamenti stessa route
                            cont += 1
                            if (cont > 1000):
                                goto.end
                            delta, deltak, deltak1 = delta_cost(routes, k, k1, i, j, d)
                            if delta < -0.001:
                                # print(f"{cont}) scambio {i} - {j} variazione {delta}")
                                delta_swap(routes, k, k1, i, j)
                                loads[k] -= requests[node]
                                loads[k1] += requests[node]
                                rcosts[k] += deltak  # deltak è negativo
                                rcosts[k1] += deltak1
                                # print(f"{cont}) scambio {routes[k]}, {routes[k1]}, delta {delta}")
                                goto.repeat
                    if (j == len(routes[k1])):
                        print("boh")
    label.end
    return routes

DISTANZE_DEPOSITI_INDEX = 20

DISTANZE_CLIENTI_DEPOSITO_INDEX = 21


# tra le colonne della matrice
# guardare quanto fa la somma => somma totale
# somma totale * 1.3 / 25

def find_min_index_in_row(dist: DataFrame, visited, i):
    min_cost = 999999
    min_index = -1
    for row in range(len(dist.iloc[i, :].values)):
        x = 0
        if dist.iloc[i, :].values[row] < min_cost and not visited[row]:
            min_cost = dist.iloc[i, :].values[row]
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

            if j == -1 or (load + demands[j] > cap):
                break

            visited[j] = True
            routes[k].append(j)
            load += demands[j]
            cost += dist.iloc[i, j]
            i = j

        cost += dist.iloc[i, 20]
        routes[k].append(20)

    return routes, cost


import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv('mat52.csv')
    scatter = pd.read_csv('customers_scatter.csv')

    distanze_depositi_df = df.iloc[:DISTANZE_DEPOSITI_INDEX + 1, :DISTANZE_DEPOSITI_INDEX + 1]

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
    plt.title('Initial')
    plt.scatter(x=scatter['x'], y=scatter['y'], )

    print(scatter.iloc[:,:].values[0]) # [ 1.    43.008 60.738]

    sols, _ = initial_solution(distanze_depositi_df, richieste_array, 10, 50)

    print(sols)

    for sol in sols:
        Xs = []
        Ys = []
        for node in sol:
            pnt = scatter.iloc[:,:].values[node-2]
            Xs.append(pnt[1])
            Ys.append(pnt[2])
        plt.plot(Xs, Ys)

    routes = local_search(
        sols,
        np.array(richieste_array),
        100, distanze_depositi_df.to_numpy()
        )

    print(routes)

    plt.figure()
    plt.title('Optimal')
    plt.scatter(x=scatter['x'], y=scatter['y'], )

    for sol in sols:
        Xs = []
        Ys = []
        for node in sol:
            pnt = scatter.iloc[:,:].values[node-2]
            Xs.append(pnt[1])
            Ys.append(pnt[2])
        plt.plot(Xs, Ys)


    plt.show(block=True)
