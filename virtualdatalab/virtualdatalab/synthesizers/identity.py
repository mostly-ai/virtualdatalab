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
        list_all = []
        new_id = 0
        for sampled_id in sampled_ids:
            sample = df_copy.loc[[sampled_id],:].reset_index().drop('id',1)
            sample['id'] = new_id
            new_id = new_id + 1
            list_all.append(sample.set_index(['id','sequence_pos']))

        df_reset = pd.concat(list_all)

        return df_reset


