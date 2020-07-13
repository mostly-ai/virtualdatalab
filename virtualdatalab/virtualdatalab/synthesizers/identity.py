import pandas as pd
import numpy as np
from virtualdatalab.synthesizers.base import BaseSynthesizer

class IdentitySynthesizer(BaseSynthesizer):
    '''

    Returns the same data used to train)

    '''
    def train(self, target_data, *args):
        super().train(target_data)
        self.target_data_ = pd.DataFrame(target_data)

    def generate(self, number_of_subjects):
        super().generate(self)

        df_copy = self.target_data_ .copy(deep=True)
        unique_ids = df_copy.index.get_level_values(0).unique()
        sampled_ids = np.random.choice(unique_ids, size=number_of_subjects, replace=True)
        grid = pd.DataFrame({'id': sampled_ids}).sort_values('id')
        grid['id_new_'] = range(0, number_of_subjects)
        dt = pd.merge(df_copy.reset_index(), grid, on='id')
        dt['id'] = dt['id_new_']
        dt = dt.drop(['id_new_'], axis=1).sort_values('id')
        dt = dt.set_index(['id', 'sequence_pos'])

        return dt


