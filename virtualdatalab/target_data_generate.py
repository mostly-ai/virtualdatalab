'''

Methods to generate data

- generate_simple_dummy

'''

import pandas as pd
import numpy as np
import string


def generate_simple_seq_dummy(number_of_subjects: int,
                              sequence_length: list(),
                              num_numeric_col: int,
                              num_categorical_col: int) -> pd.DataFrame():

    """

    Numerical -> sample from range
    Categories -> sampled letters

    :param number_of_subjects: number of subjects to appear in dummy data (identified with id col) id col starts at 0.
    :param sequence_length: list with upper bound and lower determining range of possible sequence lengths
    :param num_numeric_col: number of numeric columns
    :param num_categorical_col: number of categorical columns

    :returns panda data frame of generated simple seq data with a multi index. 1st index representing subject 2nd index
    representing sequence position

    """

    assert sequence_length[0] <= sequence_length[1], 'Lower bound must be smaller than upper bound'
    assert number_of_subjects > 0, 'Subjects must be non-zero'

    if (num_numeric_col > 0) or (num_categorical_col > 0):
        if sequence_length[0] == sequence_length[1]:
            sequence_possible_lengths = [sequence_length[0]]
        else:
            sequence_possible_lengths = np.arange(start=sequence_length[0], stop=sequence_length[1], step=1)

        random_sequence_lengths = [np.random.choice(sequence_possible_lengths) for i in range(number_of_subjects)]
        sequence_index = [sequence_exploded for sequence_length in random_sequence_lengths for sequence_exploded in
                          range(sequence_length)]
        id_index = [subject_id_list for subject_id, sequence_length in
                    zip(range(number_of_subjects), random_sequence_lengths) for subject_id_list in
                    [subject_id] * sequence_length]
        tuples = list(zip(id_index, sequence_index))
        df_dummy = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples, names=['id', 'sequence_pos']))

        for i in range(num_categorical_col):
            df_dummy[f'cat_col_{i}'] = pd.Categorical(
                list(np.random.choice(list(string.ascii_lowercase), len(df_dummy))))

        for j in range(num_numeric_col):
            df_dummy[f'num_col_{j}'] = np.random.rand(len(df_dummy))

        return df_dummy
    else:
        print("Number of columns not specified. No data is generated.")


