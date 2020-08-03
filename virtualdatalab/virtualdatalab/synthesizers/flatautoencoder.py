import torch
from torch.nn import Linear, Module, ReLU, Sequential
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import Adam

from typing import List
import numpy as np
import pandas as pd
from scipy.special import expit


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
        self.logvar = Linear(dim, embedding_dim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.mu(feature)
        logvar = self.logvar(feature)
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


def _loss_function(input_data,
                   target_data,
                   target_mapping,
                   last_index_real_data,
                   mu,
                   logvar):

    if target_mapping is None:
        recon_loss_avg = mse_loss(input_data, target_data)
    else:
        total_recon_loss = []
        last_entry = 0
        for _, info_dict in target_mapping.items():
            for col_indexing, original_indexing in info_dict.items():
                first_index = int(col_indexing.split("_")[1])
                second_index = int(col_indexing.split("_")[2])

                cat_loss = cross_entropy(input_data[:,first_index:second_index],target_data[:,original_indexing].long())
                total_recon_loss.append(cat_loss)
                last_entry = second_index

        if last_entry != last_index_real_data:
            # mixed type
            num_loss = mse_loss(input_data[:, last_entry:], target_data[:, last_entry:last_index_real_data])
            total_recon_loss.append(num_loss)

        recon_loss_avg = sum(total_recon_loss)

    KLD_avg = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / input_data.size()[0]

    return recon_loss_avg + KLD_avg


class FlatAutoEncoderSynthesizer(BaseSynthesizer):
    """
    Simple Variational Auto Encoder

    Encoder - Decoder. Data is generated from latent space sampling being fed into the trained decoder.

    Encoder layer dims are defined by hidden_size_layer_list in init. Decoder dims are reversed.

    """
    def __init__(self,
                 l2pen: float = 1e-5,
                 learning_rate:float = 1e-5,
                 hidden_size_layer_list: List = [128],
                 latent_dim: int = 64,
                 number_of_epochs: int = 20,
                 cat_threshold: float = 0.5,
                 batch_size: int = 100,
                 sample_tech: str = 'max'):

        self.l2pen = l2pen
        self.learning_rate = learning_rate
        self.hidden_size_layer_list = hidden_size_layer_list
        self.reverse_hidden_size_layer_list = hidden_size_layer_list.copy()
        self.reverse_hidden_size_layer_list.reverse()
        self.latent_dim = latent_dim
        self.number_of_epochs = number_of_epochs
        self.cat_threshold = cat_threshold
        self.batch_size = batch_size
        self.sample_tech = sample_tech

        # for formatting issues
        self._underscore_column_mapping = {}

        if torch.cuda.is_available():
            self.dev = "cuda"
        else:
            self.dev = "cpu"

        self.device = torch.device(self.dev)

        print(f"Using {self.device} for computation")

    def train(self, target_data, verbose=False):
        super().train(target_data)

        prepared_data, self._underscore_column_mapping = _clean_data(target_data)
        self.data_processor = FlatStandardOneHot()
        self.transformed_data = self.data_processor.transform(prepared_data)

        x = torch.tensor(self.transformed_data.astype(np.float32).values).to(self.device)
        train = torch.utils.data.TensorDataset(x)
        training_generator = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True,drop_last=True)

        if self.data_processor._last_index is None:
            dim_size = x.shape[1]
        else:
            dim_size = self.data_processor._last_index
        encoder = Encoder(dim_size, self.hidden_size_layer_list, self.latent_dim).to(self.device)

        self.decoder_ = Decoder(self.latent_dim, self.reverse_hidden_size_layer_list, dim_size).to(self.device)
        optimizer = Adam(list(encoder.parameters()) + list(self.decoder_.parameters()),lr=self.learning_rate,weight_decay=self.l2pen)

        for epoch in range(self.number_of_epochs):
            loss_collection = []
            for batch_idx, batch_sample in enumerate(training_generator):
                optimizer.zero_grad()
                batch_sample = batch_sample[0].to(self.device)
                mu, std, logvar = encoder(batch_sample[:, :dim_size])
                # reparametrize trick
                eps = torch.randn_like(std)
                emb = eps * std + mu
                reconstruction = self.decoder_(emb)
                # reconstruction from model
                # batch_sample original data
                # target_mapping needed for loss
                loss = _loss_function(reconstruction,
                                      batch_sample,
                                      self.data_processor._category_idx_real_data,
                                      self.data_processor._last_index,
                                      mu,
                                      logvar)
                loss_collection.append(loss.item())
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch: {epoch} // Average Loss: {np.average(loss_collection)}")
        return x

    def generate(self, number_of_subjects):
        super().generate(self)
        self.decoder_.eval()

        sample_latent = torch.randn(number_of_subjects, self.latent_dim).to(self.device)

        gd_torch = self.decoder_(sample_latent)
        if self.dev == 'cuda':
            gd_torch = gd_torch.cpu()
        df_gd = pd.DataFrame(gd_torch.detach().numpy())
        df_gd.columns = self.transformed_data.columns[:self.data_processor._last_index]
        df_gd['id'] = range(0, len(df_gd))
        df_gd.set_index('id', inplace=True)
        self.raw_generated_data = df_gd
        generated_data = self.data_processor.inverse_transform(df_gd, self.sample_tech).sort_index()

        generated_data = generated_data.rename(columns=self._underscore_column_mapping)

        return generated_data