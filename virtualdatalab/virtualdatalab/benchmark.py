"""

Compare + Benchmark

Compare - calculate benchmark function for one type of metric
benchmark - compare multiple synthesizers over multiple datasets

"""
from typing import Type,List,Callable,Dict
from pandas import DataFrame
import numpy as np
import pandas as pd

import time

from virtualdatalab.metrics import compare
from virtualdatalab.datasets.loader import load_cdnow,load_berka,load_mlb
from virtualdatalab.logging import getLogger

# Get a Logger for this module
LOGGER = getLogger(__name__)

def benchmark(syntheizers_classes:List[Type],
              datasets:Dict[str,DataFrame]={},
              verbose=False,
              log=True):
    """
    One function to rule them all.


    Train , generate and analyze specific datasets for a given set of synthesizer classes.

    Synthetic dataset

    Custom datasets must be passed through encoded_datasets, otherwise CDNOW dataset is used

    :param syntheizers_classes: list of instance of synthesizer classes
    :param datasets: dict keys: dataset name dict values : pandas dataframe in common data format
    :param verbose: verbose benchmarking

    :returns: table of results or averaged

    """
    if len(datasets) == 0:
        # put both

        cdnow = load_cdnow()
        berka = load_berka()
        mlb = load_mlb()
        datasets = {
            'cdnow':cdnow,
            'berka':berka,
            'mlb':mlb
        }

    results_list = []
    for syntheizers_class in syntheizers_classes:
        syn_name = syntheizers_class.__class__.__name__
        if verbose:
            print(f"Evaluating synthesizer {syn_name}")
        if log:
            LOGGER.info(f"Evaluating synthesizer {syn_name}")
            
        for dataset_name, dataset in datasets.items():
            try:
                if verbose:
                    print(f"With dataset {dataset_name}")
                if log:
                    LOGGER.info(f"With dataset {dataset_name}")

                target_data = dataset

                start = time.time()
                # memory consumption
                # writing a log file
                #  ｡ﾟ☆: *.☽ .* :☆ﾟ
                syntheizers_class.train(target_data)
                target_data_class = target_data
                #  ｡ﾟ☆: *.☽ .* :☆ﾟ
                end = time.time()
                diff = np.round(end - start)
                if verbose:
                    print(f"Training took {diff} seconds")
                if log:
                    LOGGER.info(f"Training took {diff} seconds")

                start = time.time()
                #  ｡ﾟ☆: *.☽ .* :☆ﾟ

                #synthetic_data = syntheizers_class.generate(len(target_data))
                sample_size = np.min([target_data.index.get_level_values('id').nunique(),10000])
                LOGGER.info(f"Generation of {sample_size}")
                synthetic_data = syntheizers_class.generate(int(sample_size))

                #  ｡ﾟ☆: *.☽ .* :☆ﾟ
                end = time.time()
                diff = np.round(end - start)
                if verbose:
                    print(f"Generating took {diff} seconds")
                if log:
                    LOGGER.info(f"Generating took {diff} seconds")

                start = time.time()
                #  ｡ﾟ☆: *.☽ .* :☆ﾟ

                # need to compare exactly what the generator sees
                compare_vals = compare(target_data_class, synthetic_data)
                df_results = pd.DataFrame([syn_name, dataset_name] + list(compare_vals.values())).T
                df_results.columns = ['synthesizer', 'dataset_name'] + list(compare_vals.keys())
                results_list.append(df_results)

                #  ｡ﾟ☆: *.☽ .* :☆ﾟ
                end = time.time()
                diff = np.round(end - start)
                if verbose:
                    print(f"Comparing took {diff} seconds")
                if log:
                    LOGGER.info(f"Comparing took {diff} seconds")
            except Exception as e:
                LOGGER.info(f"{syn_name} had {e}")


    df_results_all = pd.concat(results_list)

    return df_results_all