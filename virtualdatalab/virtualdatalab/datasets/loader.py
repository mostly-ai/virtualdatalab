import pandas as pd
from os.path import dirname,join

def load_cdnow():
    """
    load in CDNOW dataset

    :returns: pandas dataframe
    """
    module_path = join(dirname(__file__), 'data/')

    return pd.read_csv(f"{module_path}cdnow_len5.csv")


def load_berka():
    """

    Load BERKA dataset

    :returns: pandas dataframe
    """
    module_path = join(dirname(__file__), 'data/')

    return pd.read_csv(f"{module_path}berka_len50.csv")