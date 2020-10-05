import pandas as pd
import numpy as np
from virtualdatalab.synthesizers.base import BaseSynthesizer
from virtualdatalab.target_data_manipulation import  _generate_column_type_dictionary
from virtualdatalab.synthesizers.utils import _assign_column_type

class ShuffleSynthesizer(BaseSynthesizer):
    '''

    Returns the same data used to train)

    '''
    def train(self, target_data, *args):
        super().train(target_data)
        self.target_data_ = pd.DataFrame(target_data)

    def generate(self, number_of_subjects):
        super().generate(self)

        df_copy = self.target_data_ .copy(deep=True)
        for x in df_copy.columns:
            df_copy[[x]] = np.random.permutation(df_copy[[x]])

        unique_ids = df_copy.index.get_level_values(0).unique()
        sampled_ids = np.random.choice(unique_ids, size=number_of_subjects, replace=True)
        grid = pd.DataFrame({'id': sampled_ids}).sort_values('id')
        grid['id_new_'] = range(0, number_of_subjects)
        df = pd.merge(df_copy.reset_index(), grid, on='id')
        df['id'] = df['id_new_']
        df = df.drop(['id_new_'], axis=1).sort_values('id')
        df = df.set_index(['id', 'sequence_pos'])
        # merge caueses dataframes to lose their types
        reference = _generate_column_type_dictionary(self.target_data_)
        df = _assign_column_type(df,reference)

        return df
