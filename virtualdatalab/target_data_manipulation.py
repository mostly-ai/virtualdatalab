"""
Methods to help with data mainpulation

"""
import numpy as np
import pandas as pd
import json
from typing import List,Any
from pandas.api.types import is_numeric_dtype,is_categorical_dtype
from pandas import DataFrame,Series

def prepare_common_data_format(fp: Any,
                           cat_columns: List[str] = [],
                           num_columns: List[str] = []) -> DataFrame:
    """
    Cast cat columns into pd.Categorical. Cast num columns into numeric. Can accept parquet + csv files


    Any columns not passed in either list will be automatically casted. Anything not numeric is category.

    Set index to be id and sequence pos

    :param fp: filepath or pandas dataframe
    :param cat_columns: List of category columns
    :param num_columns: List of numeric columns

    :returns: same Dataframe with types


    """
    if type(fp) == str:
        file_extension = fp.split(".")[-1]

        if file_extension == 'csv':
            # hack so we don't have to declare types
            df = pd.read_csv(fp,low_memory=False)
        else:
            raise AttributeError("File extension not supported")
    elif type(fp) == DataFrame:
        df = fp
    else:
        raise AttributeError("FP not recognized")

    columns = list(df.columns)
    input_columns = cat_columns + num_columns

    assert 'id' in columns, "id col not in dataframe. please rename if possible"

    columns.sort()
    input_columns.sort()

    if input_columns == columns:
        cat_columns_convert = cat_columns
        num_columns_convert = num_columns
    else:
        unlabeled_columns = [col for col in columns if col not in input_columns]
        cat_columns_convert = cat_columns
        num_columns_convert = num_columns

        for unlabeled_column in unlabeled_columns:
            if is_numeric_dtype(df[unlabeled_column]):
                num_columns_convert.append(unlabeled_column)
            else:
                cat_columns_convert.append(unlabeled_column)

    for col in columns:
        if col in cat_columns_convert:
            df.loc[:, col] = pd.Categorical(df[col])
            # nan needs to be considered a category
            if df.loc[:, col].isnull().sum() != 0:
                df.loc[:, col] = df.loc[:, col].cat.add_categories("").fillna("")
        elif col in num_columns_convert:
            # throw error when nan in cols
            assert df.loc[:, col].isnull().sum() == 0, "Numeric columns contain NaN values"
            df.loc[:, col] = pd.to_numeric(df[col])

    df['sequence_pos'] = df.groupby('id').cumcount()

    return df.set_index(['id','sequence_pos'])


def encode_simple_common_data_format(dataset:DataFrame,
                                     shared_encoding_cols:List[str] = None,
                                     encoding:str = 'label') -> DataFrame:
    """

    Convert dataset into numpy array

    :param dataset: data (pd.DataFrame)
    :param encoding: encoding method (default label)
    :param shared_encoding_cols: list of columns that share the same encoding

    :return : encoded pandas dataframe
    """

    if encoding == 'label':
        category_columns = dataset.select_dtypes('category').columns
        df_cat = dataset[category_columns].copy()
        df_num = dataset[dataset.columns[~(dataset.columns.isin(category_columns))]].copy()
        encoding_dict = {}
        categories_not_shared_encoding = category_columns
        if shared_encoding_cols is not None:
            # certain columns have same categories so they share the label encoding
            categories_not_shared_encoding = [cat_col for cat_col in category_columns if cat_col not in shared_encoding_cols]

            # split out dataframe between cat cols and other
            df_cats = df_cat[shared_encoding_cols].copy()
            df_other = df_cat[categories_not_shared_encoding].copy()

            unique_cats = np.unique(dataset[category_columns].values.flatten())
            label_encoding_map = dict(zip(unique_cats, range(len(unique_cats))))
            df_cats_encoded = df_cats.apply(lambda x: x.map(label_encoding_map))
            df_cat = pd.merge(df_other, df_cats_encoded, how='inner', left_index=True, right_index=True)

            # write to encoding dict
            for cat_col in categories_not_shared_encoding:
                encoding_dict[cat_col] = label_encoding_map


        for cat_col in categories_not_shared_encoding:
            encoding_dict[cat_col] = dict(enumerate(df_cat[cat_col].cat.categories))
            df_cat.loc[:, cat_col] = df_cat[cat_col].cat.codes

        df_encoded = pd.merge(df_num, df_cat, how='inner', left_index=True, right_index=True)

        with open('encoding.json', 'w') as outfile:
            json.dump(encoding_dict, outfile)

        return df_encoded
    else:
        "encoding type not supported"

def split_train_val(dataset, val_percent = .10):
    """

    Create train and holdout set

    :param dataset: data (np.array)
    :param split_val: prop of validation split (default 90/10)

    :return train, holdout numpy array

    """

    # cast into dataframe for shuffling
    dataset = pd.DataFrame(dataset)
    dataset = np.array(dataset.reindex(np.random.permutation(dataset.index)))

    ratio = int(dataset.shape[0]*val_percent)

    train = dataset[ratio:, :]
    holdout = dataset[:ratio, :]

    return train, holdout

def _generate_column_type_dictionary(data) -> dict:
    column_types = dict(zip(data.dtypes.index, data.dtypes))
    column_type_dict = {}
    true_vector = []
    for col_name, dtype in column_types.items():
        if is_numeric_dtype(dtype):
            column_type_dict[col_name] = 'number'
            true_vector.append(True)
        elif is_categorical_dtype(dtype):
            column_type_dict[col_name] = 'category'
            true_vector.append(True)
        else:
            true_vector.append(False)
    assert sum(true_vector) == len(data.columns), " Data must only have number and category column types"
    return column_type_dict
