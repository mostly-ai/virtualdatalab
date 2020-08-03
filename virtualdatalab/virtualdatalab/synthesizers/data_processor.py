import pandas as pd
import functools
import random
from scipy.special import softmax

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

    If there is categorical variables, target values are appended at the end for cross entropy calculations
    """

    def transform(self, data):
        data_copy = data.copy(deep=True)
        dfs_to_merge = []

        df_category = data_copy.select_dtypes(include='category')

        if not df_category.empty:

            df_category_dummies = pd.get_dummies(df_category)
            df_category_wide = df_category_dummies.unstack().fillna(0)
            df_category_wide.columns = df_category_wide.columns.map(lambda x: '{}_{}'.format(x[1], x[0]))
            df_category_wide = df_category_wide.sort_index(axis=1)

            possible_sequence_lengths = list(set([int(x.split("_")[0]) for x in df_category_wide.columns]))

            category_columns = df_category.columns
            target_dataframe_dict = {}
            category_column_mapping = {}
            target_dataframe_list = {}
            first_pos = 0
            column_ordering = []
            column_idx_original = {}

            for category_column in category_columns:
                target_categories = []
                column_mapping = {}
                subset_category_columns = [x for x in df_category_wide.columns if category_column in x]
                for seq_length in possible_sequence_lengths:
                    columns = [x for x in subset_category_columns if seq_length == int(x.split("_")[0])]
                    column_ordering.extend(columns)
                    second_pos = first_pos + len(columns)
                    df_sliced = df_category_wide[columns]
                    mapping = {
                        col: idx for idx, col in enumerate(columns)
                    }
                    column_mapping[str(seq_length)] = columns
                    df_map = pd.DataFrame(df_category_wide[columns].idxmax(axis="columns").map(mapping),
                                          columns=[f"idx_{first_pos}_{second_pos}"])
                    first_pos = second_pos
                    target_categories.append(df_map)

                category_column_mapping[category_column] = column_mapping
                category_target = pd.concat(target_categories, axis=1)
                target_dataframe_list[category_column] = category_target
                column_idx_original[category_column] = [[pos, name] for pos, name in enumerate(category_target.columns)]

            self._category_column_mapping = category_column_mapping
            # needed for loss

            dfs_to_merge.append(df_category_wide.apply(lambda x: pd.Categorical(x))[column_ordering])

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
        self.idx_mapping = {idx: self.column_mapping[x] for idx, x in enumerate(transformed_data.columns)}
        self._last_index = None
        self._category_idx_real_data = None

        # data will need to have targets for loss
        if not df_category.empty:
            self._last_index = transformed_data.shape[1]
            self._category_idx_real_data = {}
            end_idx = transformed_data.shape[1]
            for cat_col_name, cat_target_dataframe in target_dataframe_list.items():
                transformed_data = pd.concat([transformed_data, cat_target_dataframe], axis=1)
                pos_info = column_idx_original[cat_col_name]
                self._category_idx_real_data[cat_col_name] = {name: pos + end_idx for pos, name in pos_info}
                end_idx = end_idx + cat_target_dataframe.shape[1]

        return transformed_data

    def inverse_transform(self, transformed_data, categorical_selection):
        to_merge_all = []

        types_to_transform = ['number', 'category']

        for type_to_transform in types_to_transform:

            transform_columns = [key for key, pair in self.column_mapping.items() if pair == type_to_transform]
            df_sample_transform = transformed_data[transform_columns]

            if not df_sample_transform.empty:

                if type_to_transform == 'number':

                    df_inv = ((df_sample_transform * self._transformer_std) + self._transformer_mean)
                    df_inv.loc[:, 'id'] = df_inv.index
                    df_melt = df_inv.melt(id_vars='id')
                    df_melt['col_name'] = df_melt['variable'].apply(lambda x: x.split("_")[0])
                    df_melt['sequence_pos'] = df_melt['variable'].apply(lambda x: x.split("_")[1])
                    df_pivot = df_melt.drop('variable', 1).pivot_table(index=['id', 'sequence_pos'], columns='col_name')
                    df_pivot.columns = [x[1] for x in df_pivot.columns]

                    to_merge_all.append(df_pivot)

                elif type_to_transform == 'category':
                    to_merge_categoricals = []
                    for category, sequence_mapping_dict in self._category_column_mapping.items():
                        categorical_values = []
                        for sequence_pos, column_mapping in sequence_mapping_dict.items():
                            df_slice = df_sample_transform[column_mapping]
                            if categorical_selection == 'max':
                                df_results = pd.DataFrame(df_slice.idxmax(axis=1), columns=['value'])
                            elif categorical_selection == 'sampling':
                                def f(x):
                                    return random.choices(df_slice.columns, weights=softmax(x))[0]

                                df_results = pd.DataFrame(df_slice.apply(lambda x: f(x), axis=1), columns=['value'])
                            else:
                                raise Exception("categorical_selection is not max or sampling")
                            df_results = df_results['value'].str.split("_", expand=True).drop(1, 1)
                            df_results.columns = ['sequence_pos', category]
                            df_results.reset_index(inplace=True)
                            df_results.set_index(['id', 'sequence_pos'], inplace=True)
                            categorical_values.append(df_results)

                        to_merge_categoricals.append(pd.concat(categorical_values))

                    if len(to_merge_categoricals) > 1:
                        df_column_cat = functools.reduce(
                            lambda left, right: pd.merge(left, right, on=['id', 'sequence_pos']), to_merge_categoricals)
                    else:
                        df_column_cat = to_merge_categoricals[0]

                    to_merge_all.append(df_column_cat)

        if len(to_merge_all) > 1:
            detransformed_data = functools.reduce(
                lambda left, right: pd.merge(left, right, on=['id', 'sequence_pos'], how='left'),
                to_merge_all)
        else:
            detransformed_data = to_merge_all[0]

        detransformed_data = _assign_column_type(detransformed_data, self.original_column_mapping)

        return detransformed_data