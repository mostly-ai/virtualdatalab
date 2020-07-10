import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from typing import List, Tuple
from pandas import DataFrame
import numpy as np
import pandas as pd
import functools

from pandas.api.types import is_categorical_dtype
from virtualdatalab.synthesizers.base import BaseSynthesizer

"""
.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

Encoder Decoder Mini Classes

.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.
"""


class _FlatEncoder(nn.Module):
    def __init__(self, input_size, hidden_state_sizes):
        super(_FlatEncoder, self).__init__()

        self.linear1 = torch.nn.Linear(input_size, hidden_state_sizes[0])
        self.linear2 = torch.nn.Linear(hidden_state_sizes[0], hidden_state_sizes[1])
        self.linear3 = torch.nn.Linear(hidden_state_sizes[1], hidden_state_sizes[2])

    #         self.linear4 = torch.nn.Linear(hidden_state_sizes[2], output_size)

    def forward(self, x):
        h_1 = self.linear1(x)
        h_1 = F.relu(h_1)
        h_2 = self.linear2(h_1)
        h_2 = F.relu(h_2)
        h_3 = self.linear3(h_2)
        return h_3


class _FlatDecoderNumeric(nn.Module):
    def __init__(self, hidden_state_sizes, output_size):
        super(_FlatDecoderNumeric, self).__init__()

        # list is reversed
        self.linear1 = torch.nn.Linear(hidden_state_sizes[0], hidden_state_sizes[1])
        self.linear2 = torch.nn.Linear(hidden_state_sizes[1], hidden_state_sizes[2])
        self.linear3 = torch.nn.Linear(hidden_state_sizes[2], output_size)

    def forward(self, x):
        h_1 = self.linear1(x)
        h_1 = F.relu(h_1)
        h_2 = self.linear2(h_1)
        h_2 = F.relu(h_2)
        out = self.linear3(h_2)
        return out


class _FlatDecoderCat(nn.Module):
    def __init__(self, hidden_state_sizes, output_size):
        super(_FlatDecoderCat, self).__init__()

        # list is reversed
        self.linear1 = torch.nn.Linear(hidden_state_sizes[0], hidden_state_sizes[1])
        self.linear2 = torch.nn.Linear(hidden_state_sizes[1], hidden_state_sizes[2])
        self.linear3 = torch.nn.Linear(hidden_state_sizes[2], output_size)

    def forward(self, x):
        h_1 = self.linear1(x)
        h_1 = F.relu(h_1)
        h_2 = self.linear2(h_1)
        h_2 = F.relu(h_2)
        out = self.linear3(h_2)
        out = torch.sigmoid(out)
        return out


"""
.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

Flat Auto Encoder

.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.
"""


class FlatAutoEncoderSynthesizer(BaseSynthesizer):
    """
    Simple auto encoder to generate data.

    Sequences are flattened s.t that each row represents the entire sequence
    of a given user.


    Encoder architecture:
    FC = Fully Connected Linear Layer
    Input -> 128 units FC -> 64 units FC -> 32 units FC (Encoder Output)

    Decoder architecture:
    Encoder Output -> 64 Units FC -> 128 Units FC -> (If Cat : Sigmoid)

    Encoder architecture is common for numeric + categorical data. But has different loss
    functions.

    Decoder architecture differs from last layer of sigmoid functions of possibilities.

    Due to the flat table design, it w

    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 hidden_size_layer_list: List = [512,128,64],
                 number_of_epochs: int = 20,
                 sigmoid_threshold: float = 0.5,
                 latent_dim:int = 32,
                 batch_size: int = 100):

        self.learning_rate = learning_rate
        self.hidden_size_layer_list = hidden_size_layer_list
        self.number_of_epochs = number_of_epochs
        self.sigmoid_threshold = sigmoid_threshold
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        # know which models to generate
        self._data_has_categories = False
        self._data_has_numerics = False

        # for formatting issues
        self._underscore_column_mapping = {}

        if torch.cuda.is_available():
            self.dev = "cuda"
        else:
            self.dev = "cpu"
        self.device = torch.device(self.dev)

        print(f"Using {self.device} for computation")

    def train(self, target_data: DataFrame, debug: bool = False):
        """
        Train FlatAutoEncoder. Declares decoder as self.attribute.

        :params target_data: pandas Dataframe in common data format
        :params debug: used to print epoch training statements


        """
        super().train(target_data)

        # due to some formatting issues col name can't have "_"

        col_underscore = [x for x in target_data.columns if "_" in x]
        if len(col_underscore) > 0:
            target_data = target_data.copy(deep=True)
            # force replacement for formatting purposes
            target_data_new_columns = []
            unique_column_id = 0
            # {new_column: original column}
            self._underscore_column_mapping = {}
            for column in target_data.columns:
                if column in col_underscore:
                    stripped_col = column.replace("_", "")
                    new_column_name = f"{stripped_col}aa{unique_column_id}"
                    unique_column_id = unique_column_id + 1
                    target_data_new_columns.append(new_column_name)
                    self._underscore_column_mapping[new_column_name] = column
                else:
                    target_data_new_columns.append(column)

            target_data.columns = target_data_new_columns
            # raise Exception("Please replace _  in cols {}")

        def _create_wide_table(df_sample, data_type):
            target_reset = df_sample.copy(deep=True)
            if data_type == 'categorical':
                target_dummies = pd.get_dummies(target_reset)
                target_wide = target_dummies.unstack().fillna(-1)
                target_wide.columns = target_wide.columns.map(lambda x: '{}_{}'.format(x[0], x[1]))
                target_wide = target_wide.apply(lambda x: pd.Categorical(x))
            elif data_type == 'numeric':
                target_wide = target_reset.unstack().fillna(0)
                target_wide.columns = target_wide.columns.map(lambda x: '{}_{}'.format(x[0], x[1]))
            else:
                raise Exception("Data Type not supported")

            return target_wide.reset_index()

        def _train_autoencoder(wide_data: DataFrame,
                               type_of_data: str,
                               debug: bool = False) -> Tuple[nn.Module, nn.Module]:
            """
            Trains an autoencoder model based on type of data.

            :params wide_data: one row represents a sequence from a given user
            :params type_of_data:
            possible options:
            [x] numeric
            [x] categorical
            :params debug: turn on if epoch training times should be printed

            :returns: Trained Encoder + Decoder

            """

            #### replace this with self attributes

            layer_sizes = self.hidden_size_layer_list
            reverse_layers = layer_sizes.copy()
            reverse_layers.reverse()

            if type_of_data == 'numeric':
                criterion = torch.nn.MSELoss(reduction='sum')
                decoder = _FlatDecoderNumeric(reverse_layers, wide_data.shape[1])
            elif type_of_data == 'categorical':
                # flat table is multi label, since sequence is transposed
                criterion = torch.nn.MultiLabelSoftMarginLoss()
                decoder = _FlatDecoderCat(reverse_layers, wide_data.shape[1])

            encoder = _FlatEncoder(wide_data.shape[1], layer_sizes)

            # send to gpu, if avaliable
            encoder = encoder.to(self.device)
            decoder = decoder.to(self.device)

            encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.learning_rate)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=self.learning_rate)

            # Set up DataLoader for batch processing
            x = torch.tensor(wide_data.astype(np.float32).values)
            train = torch.utils.data.TensorDataset(x)
            training_generator = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True,
                                                             num_workers=6)

            if debug:
                print(f"Training for {type_of_data}")

            for epoch in range(self.number_of_epochs):

                loss_collection = []
                for batch_idx, batch_sample in enumerate(training_generator):
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()

                    # data generator yields a list
                    batch_sample = batch_sample[0].to(self.device)

                    encoder_output = encoder(batch_sample)
                    decoder_output = decoder(encoder_output)

                    # y is target data
                    loss = criterion(decoder_output, batch_sample)

                    loss.backward()

                    # exploding gradients due to very sparse matrix
                    # we clip it to keep it stable, accepting the risk of introducing additional bias
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=10, norm_type=2)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=10, norm_type=2)

                    encoder_optimizer.step()
                    decoder_optimizer.step()

                    loss_collection.append(loss.item())
                if debug:
                    print(f"Epoch Number:{epoch}// Average Batch Loss: {np.average(loss_collection)}")

            # we only need decoder
            return decoder

        category_columns = [col for col in target_data.columns if is_categorical_dtype((target_data[col]))]
        numeric_columns = set(target_data.columns) - set(category_columns)

        target_cat = target_data[category_columns]
        target_num = target_data[numeric_columns]

        target_wide_cat = pd.DataFrame()
        target_wide_numeric = pd.DataFrame()

        if len(category_columns) > 0:
            self._data_has_categories = True
            target_wide_cat = _create_wide_table(target_cat, 'categorical')
            self.decoder_cat_ = _train_autoencoder(target_wide_cat.drop("id", 1), 'categorical', debug)

        if len(numeric_columns) > 0:
            self._data_has_numerics = True
            target_wide_numeric = _create_wide_table(target_num, 'numeric')
            self._target_data_std = target_wide_numeric.drop("id", 1).mean()
            self._target_data_mean = target_wide_numeric.drop("id", 1).std()
            target_wide_numeric = ((target_wide_numeric - self._target_data_mean) / self._target_data_std)
            self.decoder_num_ = _train_autoencoder(target_wide_numeric.drop("id", 1), 'numeric', debug)

        target_data = pd.merge(target_wide_cat, target_wide_numeric, left_on='id', right_on='id')
        # save attribute to access columns
        self._target_data_wide = target_data

    def generate(self, number_of_subjects: int):
        """
        Data is generated by feeding a random noise matrix to the numeric and categorical
        encoder


        """

        super().generate(self)

        def _convert_wide_table(to_convert: DataFrame) -> DataFrame:
            """

            Transform wide table back to long table. For metric usage.


            :params to_convert: wide DataFrame

            :returns: long DataFrame id, seq

            """

            category_columns = [col for col in self._target_data_wide.columns if
                                is_categorical_dtype((self._target_data_wide[col]))]
            # id col counts as numeric so we manually drop it out
            numeric_columns = set(self._target_data_wide.drop("id", 1).columns) - set(category_columns)

            df_cat_all = pd.DataFrame()
            df_num_all = pd.DataFrame()

            if len(category_columns) > 0:

                long_category_columns = [x.split("_")[0] for x in category_columns]

                category_dataframe = pd.DataFrame({"original_col_name": long_category_columns,
                                                   "generated_col_name": category_columns})

                unique_categories = category_dataframe['original_col_name'].unique()

                all_cat_list = []
                # lots of if conditions because the presence
                for unique_category in unique_categories:
                    categories_to_loop = category_dataframe[category_dataframe['original_col_name'] == unique_category][
                        "generated_col_name"].values
                    cat_list = []
                    for column in categories_to_loop:
                        # assuming col name structure : cds_1_0
                        split = column.split("_")
                        col_name = split[0]
                        category = split[1]
                        sequence_pos = split[-1]

                        df_copy = to_convert[['id', column]].copy(deep=True)
                        df_copy.columns = ['id', 'val']
                        df_copy.loc[:, col_name] = category
                        df_copy.loc[:, 'sequence_pos'] = int(sequence_pos)

                        if not df_copy[df_copy['val'] != 0].empty:
                            cat_list.append(df_copy[df_copy['val'] != 0].drop('val', 1))

                    if len(cat_list) > 0:
                        df_cat_seq = pd.concat(cat_list)
                        all_cat_list.append(df_cat_seq)

                if len(all_cat_list) > 0:
                    # possible that sequence positions since the sample is a multi label
                    df_cat_all = functools.reduce(lambda left, right: pd.merge(left, right, on=['id', 'sequence_pos']),
                                                  all_cat_list)
            if len(numeric_columns):
                numeric_dataframe = pd.DataFrame({"original_col_name": [x.split("_")[0] for x in numeric_columns],
                                                  "generated_col_name": list(numeric_columns)})
                unique_numeric_columns = numeric_dataframe['original_col_name'].unique()

                all_num_cols = []
                for unique_numeric_column in unique_numeric_columns:
                    numeric_sequence_columns = \
                        numeric_dataframe[numeric_dataframe['original_col_name'] == unique_numeric_column][
                            "generated_col_name"].values

                    num_list = []
                    for column in numeric_sequence_columns:
                        # assuming col name structure amt_207
                        split = column.split("_")
                        col_name = split[0]
                        sequence_pos = split[1]

                        df_copy = to_convert[['id', column]].copy(deep=True)
                        df_copy.columns = ['id', col_name]
                        df_copy.loc[:, 'sequence_pos'] = int(sequence_pos)

                        # demean
                        # self attributes act as lookup table
                        df_copy.loc[:, col_name] = (df_copy.loc[:, col_name] * self._target_data_std.loc[column]) + \
                                                   self._target_data_mean.loc[column]

                        num_list.append(df_copy)

                    df_num_seq = pd.concat(num_list)
                    all_num_cols.append(df_num_seq)

                df_num_all = functools.reduce(lambda left, right: pd.merge(left, right, on=['id', 'sequence_pos']),
                                              all_num_cols)

            # guarantee unique sequence pos per user
            merged_joined = pd.merge(df_cat_all, df_num_all, how='inner', left_on=['id', 'sequence_pos'],
                                     right_on=['id', 'sequence_pos'])

            # tidy up
            merged_joined = merged_joined[
                ['id', 'sequence_pos'] + list(set(merged_joined.columns) - set(['id', 'sequence_pos']))].sort_values(
                ['id', 'sequence_pos']).reset_index(drop=True).set_index(['id', 'sequence_pos'])

            # recast to category
            if len(all_cat_list) > 0:
                merged_joined.loc[:, category_dataframe['original_col_name'].unique()] = merged_joined.loc[:,
                                                                                         category_dataframe[
                                                                                             'original_col_name'].unique()].astype(
                    'category')

            merged_joined = merged_joined[~merged_joined.index.duplicated()]

            return merged_joined

        # last hidden size layer
        random_noise = torch.randn(number_of_subjects, self.hidden_size_layer_list[-1]).to(self.device)

        generated_data_after_prob = pd.DataFrame()
        generated_data_num = pd.DataFrame()

        if self._data_has_categories == True:
            generated_data_cat = self.decoder_cat_(random_noise)

            # sigmoid function
            sigmoid_threshold = 0.3

            if self.dev == 'cuda':
                # must send tensor to cpu first
                generated_data_cat = generated_data_cat.cpu()

            generated_data_after_prob = pd.DataFrame(np.where(generated_data_cat >= sigmoid_threshold, 1, 0))

        if self._data_has_numerics == True:
            generated_data_num_prelim = self.decoder_num_(random_noise)
            if self.dev == 'cuda':
                generated_data_num_prelim = generated_data_num_prelim.cpu()

            generated_data_num = pd.DataFrame(generated_data_num_prelim.detach().numpy())

        generated_data_wide = pd.concat([generated_data_after_prob, generated_data_num], axis=1)
        generated_data_wide.columns = self._target_data_wide.drop("id", 1).columns

        generated_data_wide['id'] = range(number_of_subjects)

        generated_data = _convert_wide_table(generated_data_wide)

        # map back to original column names

        generated_data.columns = [
            self._underscore_column_mapping[column] if column in self._underscore_column_mapping.keys() else column for
            column in generated_data.columns]

        # return generated_data_wide
        return generated_data
