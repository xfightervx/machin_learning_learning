import pandas as pd

def load_wine_dataset():
        df = pd.read_csv('data processing/wine.data',header=None)
        return df

