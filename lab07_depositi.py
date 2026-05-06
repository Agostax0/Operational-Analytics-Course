DISTANZE_DEPOSITI_INDEX=20

DISTANZE_CLIENTI_DEPOSITO_INDEX=21

# tra le colonne della matrice
# guardare quanto fa la somma => somma totale
# somma totale * 1.3 / 25

import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('mat52.csv')

    print(df)

    distanze_depositi_df = df.iloc[:DISTANZE_DEPOSITI_INDEX, :DISTANZE_DEPOSITI_INDEX]

    print(distanze_depositi_df[:].sum() * 1.3 / 25)


