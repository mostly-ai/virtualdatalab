import pandas as pd
import functools

from virtualdatalab.target_data_manipulation import _generate_column_type_dictionary
from virtualdatalab.synthesizers.utils import _assign_column_type

class DataProcessor:

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, transformed_data):
        raise NotImplementedError


class FlatStandardOneHot(DataProcessor):
    """

    Flatten a sequence table s.t that one row represents a user

    | id | sequence_pos | col_1 | col_2 |
    |----|--------------|-------|-------|
    | 1  | 1            | a     | 10    |
    | 1  | 2            | b     | 20    |
    | 2  | 1            | c     | 30    |
    | 2  | 2            | d     | 40    |

    into

    | id | col_1_1 | col_1_2 | col_2_1 | col_2_2 |
    |----|---------|---------|---------|---------|
    | 1  | a       | b       | 10      | 20      |
    | 2  | c       | d       | 30      | 40      |

    Standardize numeric columns (subtract mean and divide by std)
    One hot encoding for categorical columns
    """

    def transform(self, data):
        data_copy = data.copy(deep=True)
        dfs_to_merge = []

        df_category = data_copy.select_dtypes(include='category')

        if not df_category.empty:
            df_category_dummies = pd.get_dummies(df_category)
            df_category_wide = df_category_dummies.unstack().fillna(-1)
            df_category_wide.columns = df_category_wide.columns.map(lambda x: '{}_{}'.format(x[0], x[1]))
            df_category_wide = df_category_wide.apply(lambda x: pd.Categorical(x))
            dfs_to_merge.append(df_category_wide)

        df_numerics = data_copy.select_dtypes(include='number')

        if not df_numerics.empty:
            df_numerics_wide = df_numerics.unstack().fillna(0)
            df_numerics_wide.columns = df_numerics_wide.columns.map(lambda x: '{}_{}'.format(x[0], x[1]))
            self._transformer_mean = df_numerics_wide.mean()
            self._transformer_std = df_numerics_wide.std()
            df_numerics_wide_standard = (df_numerics_wide - self._transformer_mean) / self._transformer_std
            dfs_to_merge.append(df_numerics_wide_standard)

        if len(dfs_to_merge) > 1:
            transformed_data = functools.reduce(lambda left, right: pd.merge(left, right, on=['id']), dfs_to_merge)
        else:
            transformed_data = dfs_to_merge[0]

        self.original_column_mapping = _generate_column_type_dictionary(data)
        self.column_mapping = _generate_column_type_dictionary(transformed_data)
        # needed for loss
        self.idx_mapping = {idx: self.column_mapping[x] for idx, x in enumerate(transformed_data.columns)}

        return transformed_data

    def inverse_transform(self, transformed_data):
        data_copy_wide = transformed_data.copy(deep=True)
        data_copy_wide = _assign_column_type(data_copy_wide, self.column_mapping)
        dfs_to_merge = []


        types_to_transform = ['number', 'category']

        for type_to_transform in types_to_transform:
            df_sample_tranform = data_copy_wide.select_dtypes(include=type_to_transform)

            if not df_sample_tranform.empty:

                df_column = pd.DataFrame({"original_col_name": [x.split("_")[0] for x in df_sample_tranform.columns],
                                          "generated_col_name": df_sample_tranform.columns})

                unique_columns = df_column['original_col_name'].unique()

                concat_column_list = []

                for unique_column in unique_columns:
                    """
                    ex 
                    unique_columns = cd
                    sub_column = cd_0_1
                    """
                    sub_columns = df_column[df_column['original_col_name'] == unique_column][
                        "generated_col_name"].values
                    df_one_column_list = []

                    for sub_column in sub_columns:
                        split = sub_column.split("_")
                        col_name = split[0]

                        # ✧･ﾟ: *✧･ﾟ:* *:･ﾟwday_nan_4✧*:･ﾟ✧
                        if type_to_transform == 'category':
                            category = split[1]
                            sequence_pos = split[-1]
                        elif type_to_transform == 'number':
                            sequence_pos = split[1]
                        else:
                            raise Exception('Type not recognized')
                        # ✧･ﾟ: *✧･ﾟ:* *:･ﾟ✧*:･ﾟ✧

                        mini_copy = df_sample_tranform[[sub_column]].copy(deep=True)
                        mini_copy.loc[:, 'sequence_pos'] = int(sequence_pos)

                        # ✧･ﾟ: *✧･ﾟ:* *:･ﾟ✧*:･ﾟ✧
                        if type_to_transform == 'category':
                            mini_copy.columns = ['val', 'sequence_pos']
                            mini_copy.loc[:, col_name] = category

                            if not mini_copy[mini_copy['val'] != 0].empty:
                                df_one_column_list.append(mini_copy[mini_copy['val'] != 0].drop('val', 1))
                            else:
                                df_one_column_list.append(mini_copy.drop('val',1))

                        elif type_to_transform == 'number':
                            mini_copy.columns = [col_name, 'sequence_pos']
                            mini_copy.loc[:, col_name] = (mini_copy.loc[:, col_name] * self._transformer_std.loc[
                                sub_column]) + \
                                                         self._transformer_mean.loc[sub_column]
                            df_one_column_list.append(mini_copy)
                        else:
                            raise Exception(
                                'Type not recognized')

                        # ✧･ﾟ: *✧･ﾟ:* *:･ﾟ✧*:･ﾟ✧

                    df_one_columns = pd.concat(df_one_column_list)
                    concat_column_list.append(df_one_columns)

                if len(concat_column_list) > 1:
                    df_column_all = functools.reduce(
                        lambda left, right: pd.merge(left, right, on=['id', 'sequence_pos']), concat_column_list)
                else:
                    df_column_all = concat_column_list[0]

                dfs_to_merge.append(df_column_all)

        if len(dfs_to_merge) > 1:
            detransformed_data = functools.reduce(lambda left, right: pd.merge(left, right, on=['id', 'sequence_pos'],how='left'),
                                                  dfs_to_merge)
        else:
            detransformed_data = dfs_to_merge[0]

        detransformed_data = detransformed_data.reset_index().set_index(['id', 'sequence_pos'])
        detransformed_data = detransformed_data[~detransformed_data.index.duplicated()]
        # types are lost from multiple merges
        detransformed_data = _assign_column_type(detransformed_data, self.original_column_mapping)

        return detransformed_data.sort_index()