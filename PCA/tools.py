import pandas as pd

def load_wine_dataset():
        df = pd.read_csv('PCA/wine.data',header=None)
        return df

