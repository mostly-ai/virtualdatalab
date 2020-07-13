import pandas as pd
from os.path import dirname,join
from virtualdatalab.target_data_manipulation import prepare_common_data_format

def load_cdnow():
    """
    load in CDNOW dataset

    :returns: pandas dataframe
    """
    module_path = join(dirname(__file__), 'data/')

    return prepare_common_data_format(pd.read_csv(f"{module_path}cdnow_len5.csv"))


def load_berka():
    """

    Load BERKA dataset

    :returns: pandas dataframe
    """
    module_path = join(dirname(__file__), 'data/')

    return prepare_common_data_format(pd.read_csv(f"{module_path}berka_len10.csv"))