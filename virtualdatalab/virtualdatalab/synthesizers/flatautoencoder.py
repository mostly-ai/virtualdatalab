import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import multilabel_soft_margin_loss, mse_loss
from torch.optim import Adam

import pandas as pd

from typing import List, Tuple
from pandas import DataFrame
import numpy as np
import pandas as pd
import functools

from pandas.api.types import is_categorical_dtype


from virtualdatalab.synthesizers.base import BaseSynthesizer
from virtualdatalab.synthesizers.data_processor import FlatStandardOneHot


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.mu = Linear(dim, embedding_dim)
        self.std = Linear(dim, embedding_dim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.mu(feature)
        logvar = self.std(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [
                Linear(dim, item), ReLU()
            ]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


def _clean_data(data):
    """
    _ is used as splitting character for turning long into wide.
    Thus we must get rid of any columns with this _ and replace with something different
    """
    col_underscore = [x for x in data.columns if "_" in x]
    data_copy = data.copy(deep=True)
    if len(col_underscore) > 0:
        # force replacement for formatting purposes
        data_new_columns = []
        unique_column_id = 0
        # {new_column: original column}
        underscore_column_mapping = {}
        for column in data_copy.columns:
            if column in col_underscore:
                new_column_name = f"{column.replace('_', '')}aa{unique_column_id}"
                unique_column_id = unique_column_id + 1
                data_new_columns.append(new_column_name)
                underscore_column_mapping[new_column_name] = column
            else:
                data_new_columns.append(column)

        data_copy.columns = data_new_columns
        return data_copy, underscore_column_mapping
    else:
        return data_copy, {}


def _loss_function(input_data, target_data, mu, logvar, max_cat_idx):

    cat_loss = multilabel_soft_margin_loss(input_data[:, :max_cat_idx], target_data[:, :max_cat_idx])
    num_loss = mse_loss(input_data[:, max_cat_idx:], target_data[:, max_cat_idx:])

    recon_loss_avg = cat_loss + num_loss
    KLD_avg = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / input_data.size()[0]

    return recon_loss_avg + KLD_avg


class FlatAutoEncoderSynthesizer(BaseSynthesizer):
    def __init__(self,
                 learning_rate: float = 0.001,
                 hidden_size_layer_list: List = [512, 128, 64],
                 latent_dim: int = 32,
                 number_of_epochs: int = 20,
                 cat_threshold: float = -2,
                 batch_size: int = 100):

        self.learning_rate = learning_rate
        self.hidden_size_layer_list = hidden_size_layer_list
        self.reverse_hidden_size_layer_list = hidden_size_layer_list.copy()
        self.reverse_hidden_size_layer_list.reverse()
        self.latent_dim = latent_dim
        self.number_of_epochs = number_of_epochs
        self.cat_threshold = cat_threshold
        self.batch_size = batch_size

        # for formatting issues
        self._underscore_column_mapping = {}

        if torch.cuda.is_available():
            self.dev = "cuda"
        else:
            self.dev = "cpu"

        self.device = torch.device(self.dev)

        print(f"Using {self.device} for computation")

    def train(self, target_data, verbose=True):
        super().train(target_data)

        prepared_data, self._underscore_column_mapping = _clean_data(target_data)
        self.data_processor = FlatStandardOneHot()
        self.transformed_data = self.data_processor.transform(prepared_data)

        index_mapping = self.data_processor.idx_mapping

        x = torch.tensor(self.transformed_data.astype(np.float32).values).to(self.device)
        train = torch.utils.data.TensorDataset(x)
        training_generator = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True)

        encoder = Encoder(x.shape[1], self.hidden_size_layer_list, self.latent_dim).to(self.device)

        self.decoder_ = Decoder(self.latent_dim, self.reverse_hidden_size_layer_list, x.shape[1]).to(self.device)
        optimizer = Adam(list(encoder.parameters()) + list(self.decoder_.parameters()), weight_decay=self.learning_rate)

        self.max_cat_idx = np.max([key for key, val in index_mapping.items() if val == 'category']) + 1

        for epoch in range(self.number_of_epochs):
            loss_collection = []
            for batch_idx, batch_sample in enumerate(training_generator):
                optimizer.zero_grad()
                batch_sample = batch_sample[0].to(self.device)
                mu, std, logvar = encoder(batch_sample)
                # reparametrize trick
                eps = torch.randn_like(std)
                emb = eps * std + mu
                reconstruction = self.decoder_(emb)
                loss = _loss_function(reconstruction, batch_sample, mu, logvar, self.max_cat_idx)
                loss_collection.append(loss.item())
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch epoch: {epoch} // Average Loss: {np.average(loss_collection)}")

    def generate(self, number_of_subjects):
        super().generate(self)
        self.decoder_.eval()

        sample_latent = torch.randn(number_of_subjects, self.latent_dim).to(self.device)

        gd_torch = self.decoder_(sample_latent)
        if self.dev == 'cuda':
            gd_torch = gd_torch.cpu()
        gd_raw = pd.DataFrame(gd_torch.detach().numpy())
        df_gd = pd.concat([pd.DataFrame(np.where(gd_raw.iloc[:, :self.max_cat_idx] >= self.cat_threshold, 1, 0)),
                           gd_raw.iloc[:, self.max_cat_idx:]], axis=1)
        df_gd.columns = self.transformed_data.columns
        df_gd['id'] = range(0, len(df_gd))
        df_gd.set_index('id', inplace=True)
        generated_data = self.data_processor.inverse_transform(df_gd).sort_index()

        generated_data = generated_data.rename(columns=self._underscore_column_mapping)

        return generated_data