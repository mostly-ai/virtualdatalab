''''


Provide methods that calculate the distance between two (encoded) sequential datasets:

Similarity to Target:
* univarate total variation distance
* bivariate total variation distance
* auto-correlation total variation distance

Privacy Metrics:
* distance to closest records - quantiles
* nearest neighbor distance ratio - quantiles



'''
import pandas as pd
import numpy as np
from itertools import product
from pandas import DataFrame,Series
from typing import List,Tuple,Dict, Callable
import scipy.stats as ss
from sklearn.neighbors import NearestNeighbors

from virtualdatalab.synthesizers.utils import check_common_data_format
from virtualdatalab.target_data_manipulation import _generate_column_type_dictionary

from virtualdatalab.cython.cython_metric import mixed_distance

"""
<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>
<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>

Data Preprocessing Functions

[x] _generate_column_type_dictionary
[x] _sample_data
[x] _flatten_table
[x] _bin_data
[x] _prepare_data_for_privacy_metrics

<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>
<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>

"""

def _sample_one_event(data: DataFrame) -> DataFrame:

    """
    Randomly sample one record for each id.
    """    # determine sequence length for each id
    seq_lens = data.reset_index().groupby('id').size()    # randomly draw a sequence_pos for each id
    draws = pd.DataFrame({'sequence_pos': np.floor(np.random.rand(len(seq_lens)) * seq_lens).astype(int)}).reset_index()    # inner join with provided dataframe to filter to drawn records
    out = pd.merge(draws, data, on=['id', 'sequence_pos']).drop(columns='sequence_pos').set_index('id')
    return out

def _sample_two_events(data: DataFrame) -> DataFrame:
    """
    Randomly sample two consecutive records for each id, and flatten into one row per id.
    """    # determine sequence length for each id
    seq_lens = data.reset_index().groupby('id').size()    # filter to those ids that have at least two records
    seq_lens = seq_lens[seq_lens>=2]    # randomly draw a sequence_pos (excl last item) for each id
    draws1 = pd.DataFrame({'sequence_pos': np.floor(np.random.rand(len(seq_lens)) * (seq_lens-1)).astype(int)})
    draws2 = draws1 + 1    # inner join with provided dataframe to filter to drawn records
    item1 = pd.merge(draws1.reset_index(), data, on=['id', 'sequence_pos']).drop(columns='sequence_pos')
    item2 = pd.merge(draws2.reset_index(), data, on=['id', 'sequence_pos']).drop(columns='sequence_pos')
    out = pd.merge(item1, item2, on='id', suffixes=['_0', '_1']).set_index('id')
    return out

def _bin_data(target_col,
              syn_col,
              col_type: str,
              number_of_bins: int):
    """
    Bin Wide Data

    Numerics must have 10 distinct bins.

    Category is left alone unless categories exceeds a given cardinality (100)


    """
    if col_type == 'number':
        filled_bins = 0
        q = number_of_bins
        while filled_bins < number_of_bins:
            binned_target, bins = pd.qcut(target_col,q=q,retbins=True,duplicates='drop',precision=1)
            filled_bins = sum(binned_target.value_counts()!=0)
            q += 1
        # if synthetic has categories not in target, na value will appear
        # we drop this value
        binned_syn = pd.cut(syn_col, bins=bins, include_lowest=True,precision=1).dropna()
    elif col_type == 'category':
        cardinality = target_col.nunique()
        if cardinality >= 100:
            cut_off= 10
            top_cat = target_col.value_counts().sort_values(ascending=False)[:cut_off-1].index
            other = target_col.value_counts().sort_values(ascending=False)[cut_off-1:].index
            first_dictionary = {cat: cat for cat in cut_off}
            first_dictionary.update({cat: '*' for cat in other})
            binned_target = target_col.map(first_dictionary)
            # if synthetic has categories not in target, na value will appear
            # we drop this value
            binned_syn = syn_col.map(first_dictionary).dropna()
        else:
            binned_target = target_col
            binned_syn = syn_col
    else:
        raise Exception("Col type not recognized")

    return binned_target, binned_syn

def _bin_looped(target,synthetic,column_dictionary,number_of_bins):
    t = pd.DataFrame()
    s = pd.DataFrame()
    for col, col_type in column_dictionary.items():
        binned_target, binned_syn = _bin_data(target[col], synthetic[col], col_type, number_of_bins)
        t[col] = binned_target
        s[col] = binned_syn
    return t,s

def _flatten_table(data: DataFrame, column_type_dictionary: dict) -> DataFrame:
    """

    Flatten a table from long format into wide format.

    Long format

    | id | col_1 | col_2 | record_pos | sequence_pos |
    |----|-------|-------|------------|--------------|
    | 1  | a     | 10    | 1          | 3            |
    | 1  | b     | 20    | 2          | 4            |
    | 2  | c     | 30    | 1          | 0            |
    | 2  | d     | 40    | 2          | 1            |

    Wide format

    | id | col_a_1 | col_a_2 | col_b_1 | col_b_2 |
    |----|---------|---------|---------|---------|
    | 1  | a       | b       | 10      | 20      |
    | 2  | c       | d       | 30      | 40      |

    This is used as a utility function to do data preprocessing for metric calculations.

    Note: Sequence pos and record pos is dropped from returned dataframe.

    :param data: Pandas Dataframe
    :param column_type_dictionary: dict mapping columns to types. Note, pandas.pivot() coerces types so we need to reassign them.

    :returns: wide Pandas DataFrame

    """

    to_exclude = ['sequence_pos', 'record_pos', 'id']
    col_to_melt = [x for x in data.columns if x not in to_exclude]
    pivot = data.pivot(index='id', columns='sequence_pos', values=col_to_melt)

    # Lots of extra steps because pandas.pivot does not preserve column type

    new_columns = ["{}_{}".format(x[0], x[1]) for x in pivot.columns]
    original_columns_family = ["{}".format(x[0]) for x in pivot.columns]
    new_columns_to_family = dict(zip(new_columns, original_columns_family))
    pivot.columns = new_columns

    for col in pivot:
        original_col_type = column_type_dictionary[new_columns_to_family[col]]
        if original_col_type == 'category':
            pivot[col] = pd.Categorical(pivot[col])
        elif original_col_type == 'number':
            pivot[col] = pd.to_numeric(pivot[col])

    return pivot


def _prepare_data_for_privacy_metrics(target_data: DataFrame,
                                      syn_data: DataFrame,
                                      column_dictionary: Dict,
                                      smoothing_factor:float) -> Tuple[DataFrame, DataFrame]:
    """
    Data preparation for privacy metrics

    For categorical, ordinal encoding based on joint set of target data and synthetic data.
    For numeric encoding, missing value are imputed with mean and standardized

    :param target_data: pandas dataframe
    :param syn_data: pandas dataframe
    :param column_dictionary: column to type mapping
    :param smoothing_factor: smoothing factor


    :returns: privacy ready target + synthetic  (pamdas DataFrame)
    """

    target_data_p = target_data.copy(deep=True)
    syn_data_p = syn_data.copy(deep=True)

    for column_name, column_type in column_dictionary.items():
        if column_type == 'category':

            target_data_p[column_name] = target_data_p[column_name].cat.codes
            syn_data_p[column_name] = syn_data_p[column_name].cat.codes

        elif column_type == 'number':
            # fill na data with mean
            target_data_p[column_name] = target_data_p[column_name].fillna(target_data_p[column_name].dropna().mean())
            syn_data_p[column_name] = syn_data_p[column_name].fillna(syn_data_p[column_name].dropna().mean())

            # standardize
            target_data_p[column_name] = (target_data_p[column_name] - target_data_p[column_name].mean()) / np.max(
                [target_data_p[column_name].std(), smoothing_factor])
            syn_data_p[column_name] = (syn_data_p[column_name] - syn_data_p[column_name].mean()) / np.max(
                [syn_data_p[column_name].std(), smoothing_factor])

        else:
            raise Exception(f'{column_type} Type not supported')

    # drop id col since it's not needed
    target_data_p = target_data_p.reset_index().drop('id',1)
    syn_data_p = syn_data_p.reset_index().drop('id',1)

    return target_data_p,syn_data_p


"""
<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>
<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>

Metric Calculation Functions

- they typically operate on a given column

[x] _uni_etvd
[x] _bi_etvd
[x] _calculate_chisq_log related to calculate_correlation
[x] _calculate_correlation
[x] _get_nn_model
[x] _calculate_dcr_nndr

<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>
<*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>

"""


def _uni_etvd(binned_target: Series,
             binned_syn: Series,
             col_type: str):
    '''
    Univariate Empirical Total Variation Distance (UETVD) is the max difference in proportions between the target col and synthetic col distribution.

    Value is bounded between 0 and 1

    Empirically it is calculated by
    1.
        i. For numeric values, cols are sorted into bins dervived from the target col
        ii.For categorical values, cols are sorted into bins, if cardinality is very large the bins will condense into top 20 categories. The last category
           will be a collection of everything past top 19.
    2. Calculate the frequency for each bin.
    3. Calculate the difference in frequency between target col and syn col
    4. Max(difference) = UETVD

    For cat columns, Steps 2-3 are executed

    :param binned_target: binned col from target data
    :param syn_col: binned col from syn data
    :param col_type: numeric or categorical
    :param number_of_bins: number of bins for numeric variable

    return UETVD (abs percent difference)

    Authored: 17.06.2020
    '''

    def prep_binning(df_binned_raw, alt_col):
        '''
        Mini func to create prop of values in each bin
        '''
        binned_counts = df_binned_raw.value_counts().reset_index()
        binned_counts.columns = ['bins', alt_col]
        binned_counts[f'{alt_col}_prop'] = (binned_counts[alt_col] / binned_counts[alt_col].sum())
        return binned_counts

    if col_type in ('category', 'number'):

        binned_target_prop = prep_binning(binned_target, 'target')
        binned_syn_prop = prep_binning(binned_syn, 'syn')

        merged_bined = pd.merge(binned_target_prop, binned_syn_prop,how='left')
        merged_bined['syn_prop'] = merged_bined['syn_prop'].fillna(0)
        merged_bined['diff'] = abs(merged_bined['target_prop'] - merged_bined['syn_prop'])

        return merged_bined['diff'].max()

    else:
        print("Unsupported type entered.")


def _bi_etvd(binned_target_r: DataFrame,
            binned_syn_r: DataFrame,
            first_col: str,
            second_col: str) -> float:
    """
    Bivariate Empirical Total Variation Distance (BIETVD) is the max difference in proportions between the target col 1 col 2 and syn col 1 col 2 distribution

    1. Bin Data according to column type
    2. Cross Tab calculate frequency
    3. Subtract target cab from syn cross tab
    4. Max difference is BIETVD

    :param binned_target_r: sliced binned target dataframe of two columns
    :param binned_syn_r: sliced binned synthetic dataframe of two columns
    :param first_col: str name of first col
    :param second_col: str name of second col

    """
    target_d = binned_target_r.copy()
    syn_d = binned_syn_r.copy()

    target_ct = pd.crosstab(target_d[first_col], target_d[second_col]) / len(target_d)
    syn_ct = pd.crosstab(syn_d[first_col], syn_d[second_col])/ len(syn_d)

    # extra work if synthetic does not have classes that is in target
    # any extra classes synthetic may have are dropped out in _bin_data

    missing_columns = [x for x in target_ct.columns if x not in syn_ct.columns]
    missing_indexes = [x for x in target_ct.index if x not in syn_ct.index]
    if missing_columns:
        for missing_column in missing_columns:
            # add missing column to category list if not existing
            if missing_column not in syn_ct.columns.categories:
                syn_ct.columns = syn_ct.columns.add_categories(missing_column)
             # add empty val
            syn_ct.loc[:, missing_column] = 0
    if missing_indexes:
        for missing_index in missing_indexes:
            # add missing column to category list if not existing
            if missing_index not in syn_ct.index.categories:
                syn_ct.index = syn_ct.index.add_categories(missing_index)
            # add empty val
            syn_ct.loc[missing_index, :] = 0

    # make sure columns + index lines up
    return np.max(abs(target_ct - syn_ct.loc[target_ct.index,target_ct.columns]).values)


def _calculate_chisq_log(
        data: DataFrame,
        columns: List[str]) -> float:
    """
    Calculate logarithmic chi square

    :param data: matrix for which calculate chi square between columns
    :param columns: column names
    :return: log chi square or -1 if density is NaN
    """

    number_unique_rows, number_unique_cols = data[columns].nunique().values

    df = (number_unique_rows - 1) * (number_unique_cols - 1)

    confusion_matrix = (
        data.groupby(list(columns))
            .count()
            .unstack(fill_value=0)
            .fillna(0)
    )
    try:
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        log_p = ss.chi2.logpdf(x=chi2, df=df)
    # can't compute for small frequencies 
    except:
        log_p = np.nan
        
    if np.isnan(log_p):
        log_p = -1

    return log_p


def _calculate_correlation(group_data, columns):
    chisq_matrix = pd.DataFrame(index=columns, columns=columns)
    # use only upper part of the matrix
    upper = np.triu_indices(chisq_matrix.values.shape[0], 0)

    # calculate log chi square
    for i, j in zip(upper[0], upper[1]):
        row = columns[i]
        col = columns[j]
        chisq_matrix[col][row] = _calculate_chisq_log(group_data, [row, col])

        if i != j:
            chisq_matrix[row][col] = chisq_matrix[col][row]

    log_chisq_correlations = np.maximum(0, np.log(-chisq_matrix.astype(float)))

    maxima = pd.DataFrame(
        np.minimum.outer(
            log_chisq_correlations.max().to_list(),
            log_chisq_correlations.max().to_list(),
        ),
        columns=columns,
        index=columns,
    )

    normalized_correlations = np.square(log_chisq_correlations.divide(maxima))

    return normalized_correlations

def _get_nn_model(train: DataFrame, cat_slice: int) -> Tuple[np.ndarray]:
    """
    Find nearest neighbors of test in train with first categoric_slice-many variables being categorical.

    :param train: train pandas dataframe
    :param cat_slice: where do category columns end

    :returns: scikit learn nearest_neighbor_model fit with train data

    """
    nearest_neighbor_model = NearestNeighbors(
        metric=lambda x, y: mixed_distance(x, y, cat_slice),
        algorithm="ball_tree",
        n_jobs=None,
    )
    nearest_neighbor_model.fit(train)

    return nearest_neighbor_model


def _calculate_dcr_nndr(target_data_privacy: DataFrame,
                        syn_data_privacy: DataFrame,
                        column_dictionary: dict,
                        smoothing_factor: int) -> Tuple[Dict, Dict]:
    """
    Function to calculate dcr and nndr. Since DCR and NNDR are related, both are calculated at the same time.

    :param target_data_privacy: privacy prepared dataset target
    :param syn_data_privacy: privacy prepared dataset synthetic
    :param column_dictionary: column to type mapping
    :param smoothing_factor: smoothing factor to avoid small division

    :returns: bins and histograms of dcr and nndr

    """

    """
    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    Variables to Set

    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    """

    # how many columns to include
    max_features = 50
    # multiplicative factor for determing sample size from target data size
    sample_ratio = 0.5
    # max sample size for querying
    max_sample_size = 10000

    # bound value to determine quantiles
    dcr_quantile = 0.95
    # how many bins should be created for privacy histograms
    privacy_number_of_bins = 20

    # col names
    dcr_col_name = 'distance_to_closest_record'
    nndr_col_name = 'nearest_neighbor_distance_ratio'

    # derive model based on sample of features

    model_columns = list(column_dictionary.keys())
    sample_feature_amount = min(len(model_columns), max_features)
    feature_columns = np.random.choice(model_columns, sample_feature_amount)

    # shift columns to put category features first for distance metric
    category_columns = [x for x in feature_columns if column_dictionary[x] == 'category']
    ordered_columns = category_columns + [x for x in feature_columns if column_dictionary[x] == 'number']
    # where do cat columns begin
    cat_slice = len(category_columns)

    target_data_privacy = target_data_privacy[ordered_columns]
    syn_data_privacy = syn_data_privacy[ordered_columns]

    assert all(target_data_privacy.columns == target_data_privacy.columns), "Train and Syn have mismatched columns"

    # split into tgt_train, tgt_query, and syn_query

    target_size = len(target_data_privacy)
    synthetic_size = len(syn_data_privacy)

    sample_size = min(max_sample_size,
                      sample_ratio * target_size,
                      synthetic_size)

    shuffled_target_train_index = list(target_data_privacy.index)
    np.random.shuffle(shuffled_target_train_index)
#     print(shuffled_target_train_index)
#     print(sample_size)

    tgt_train, tgt_query = (target_data_privacy.loc[shuffled_target_train_index[:-int(sample_size)]],
                            target_data_privacy.loc[shuffled_target_train_index[-int(sample_size):]])

    shuffled_target_syn_index = list(syn_data_privacy.index)
    np.random.shuffle(shuffled_target_syn_index)

    # can be omitted since syn_train is not needed
    # if sample_size = synthetic_size, syn_query is all syn dataset
    _, syn_query = (syn_data_privacy.loc[shuffled_target_syn_index[:-int(sample_size)]],
                    syn_data_privacy.loc[shuffled_target_syn_index[-int(sample_size):]])

    # training model
    nn_model = _get_nn_model(tgt_train, cat_slice)

    tgt_query_neighbors = nn_model.kneighbors(tgt_query, n_neighbors=2)
    syn_query_neightbors = nn_model.kneighbors(syn_query, n_neighbors=2)

    # Calculating DCR NNDR
    query_dict = {
        'target': tgt_query_neighbors,
        'syn': syn_query_neightbors
    }

    privacy_data = {}

    for label, query in query_dict.items():
        dcr = query[0][:, 0]
        nndr = query[0][:, 0] / np.maximum(query[0][:, 1], smoothing_factor)
        df_privacy = pd.DataFrame({dcr_col_name: dcr,
                                   nndr_col_name: nndr})
        privacy_data[label] = df_privacy
    # get histograms and bins

    bins = {}
    histograms = {}

    for type_ in [dcr_col_name, nndr_col_name]:
        histograms[type_] = dict()
        baseline_data = privacy_data['target'][type_].dropna()
        histograms[type_]['target'], bins[type_] = np.histogram(
            baseline_data, bins=privacy_number_of_bins, density=True
        )

        data = privacy_data['syn'][type_].dropna()
        histograms[type_]['syn'], _ = np.histogram(
            data, bins=bins[type_], density=True
        )

    # norm results
    dcr_nndr_data_norm = {key: df.copy() for key, df in privacy_data.items()}
    baseline_dcr = dcr_nndr_data_norm['target'][dcr_col_name]
    bound = np.quantile(baseline_dcr[~np.isnan(baseline_dcr)], dcr_quantile)
    for key in dcr_nndr_data_norm:
        dcr_nndr_data_norm[key][dcr_col_name] = np.where(
            dcr_nndr_data_norm[key][dcr_col_name] <= bound,
            dcr_nndr_data_norm[key][dcr_col_name] / bound,
            1,
        )

    # quantile test

    def _empirical_ci(
        sample_value: float, boot_values: List[float], alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Empirical confidence intervals from bootstrap values.
        See: https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability
        -and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
        """

        boot_diffs = [sample_value - boot_value for boot_value in boot_values]
        low_diff, high_diff = np.quantile(boot_diffs, [alpha / 2, 1 - alpha / 2])

        return sample_value - high_diff, sample_value - low_diff

    def _bootstrap_func(
        series: Series,
        function: Callable[[Series], float],
        bootstrap_kwargs: Dict,
    ) -> Tuple[float, float, float]:
        """Get the empirical full-sample bootstrap estimate for a given function with
        confidence intervals.
        """

        sample_size = len(series)
        sample_value = function(series)

        boot_values = [
            function(
                np.random.choice(
                    series,
                    sample_size,
                    replace=bootstrap_kwargs.get(bootstrap_kwargs['repeat'], True),
                )
            )
            for _ in range(bootstrap_kwargs.get(bootstrap_kwargs['repeat'],1000))
        ]

        confidence_interval_low, confidence_interval_high = _empirical_ci(
            sample_value,
            boot_values,
            alpha=bootstrap_kwargs.get(bootstrap_kwargs['alpha'], 0.05),
        )

        return sample_value, confidence_interval_low, confidence_interval_high

    def _bootstrap_quantile(
        series: Series,
        quantile: float,
        bootstrap_kwargs: Dict,
    ) -> Tuple[float, float, float]:
        """Bootstrap estimate for a quantile with confidence intervals."""

        return _bootstrap_func(
            series, lambda series_: np.quantile(series_, quantile), bootstrap_kwargs,
        )

    def _quantile_test_function(target:Series,
                                synthetic:Series)-> Dict:
        """
        * we look at a set of quantiles
        * we bootstrap each tgt quantile with confidence intervals
        * we fail the test if any of the synthetic quantiles is below the
        lower confidence bound of the corresponding tgt quantile.

        :param target: the target data for testing quantile shift
        :param synthetic: the synthetic data for testing shift
        :return: the test result dictionary

        """

        quantiles = np.linspace(0.05, 0.5, 20)

        bootstrap_kwargs = {
           "repeat":1000,
            "alpha":0.05

        }
        alpha_init = bootstrap_kwargs['alpha']
        alpha_adj = alpha_init / len(quantiles)
        bootstrap_kwargs['alpha'] = alpha_adj

        # boostrap the quantiles
        bootstrap_results = pd.DataFrame(
            index=quantiles,
            columns=[
                'synthetic',
                'target',
                'target_ci_low',
                'target_ci_high',
                'check',
            ],
        )

        for quantile in quantiles:
            bootstrap_results.loc[quantile,
                                  ['target','target_ci_low','target_ci_high']
            ] = _bootstrap_quantile(target,quantile,bootstrap_kwargs)
            bootstrap_results.loc[quantile,'synthetic'] = np.quantile(synthetic, quantile)

        bootstrap_results['check'] = (
                bootstrap_results['synthetic'] >= bootstrap_results['target_ci_low']
                                      )
        final_check = np.all(bootstrap_results['check'])
        bootstrap_results_dict = bootstrap_results.to_dict("series")

        return {
            'check':final_check,
            'details':bootstrap_results_dict
        }

    privacy_tests = {}
    checks = {}

    for privacy_type in [dcr_col_name,nndr_col_name]:
        target = dcr_nndr_data_norm['target'][privacy_type]
        synthetic = dcr_nndr_data_norm['syn'][privacy_type]
        privacy_tests[privacy_type] = _quantile_test_function(target,synthetic)
        checks[privacy_type] = privacy_tests[privacy_type]['check']

    return checks,privacy_tests


def _calculate_accuracy_metric(target_data:DataFrame,
                              synthetic_data:DataFrame,
                              metric_func_name:str,
                              summary_func_name:str='mean',
                              number_of_bins:int = 10)-> float:
    """
    Compute a single metric for a given target and synthetic data set

    Please see metrics.py for a detail breakdown of each metric.

    :param target_data: target data
    :param synthetic_data: synthetic data
    :param metric_func_name: metric function to be tested
    possible options

    accuracy:
    uni_etvd
    bi_etvd
    correlation

    :param summary_func_name: function to aggregate values
    possible options
    median
    mean
    max

    :param number_of_bins: number of bins for numeric data

    :returns: averaged result calculated by summary_func

    """
    """
    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    summary_func selection

    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    """

    dictionary_functions = {
        'median': np.median,
        'mean': np.mean,
        'max': np.max

    }

    summary_func = dictionary_functions[summary_func_name]

    """
    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    Data Processing

    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    """

    # check if data is in expected format

    assert sorted(target_data.columns) == sorted(synthetic_data.columns), "Target and Synthetic have different columns"

    original_column_type_dictionary = _generate_column_type_dictionary(target_data)

    """
    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    Metric Calculation

    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.
    """

    if metric_func_name == 'uni_etvd':
        '''
        Data Prep
        '''
        sample_one_target = _sample_one_event(target_data)
        sample_one_synthetic = _sample_one_event(synthetic_data)

        one_t_binned, one_s_binned = _bin_looped(sample_one_target,
                                                 sample_one_synthetic,
                                                 original_column_type_dictionary,
                                                 number_of_bins)

        univariate_values = []
        for col_name, col_type in original_column_type_dictionary.items():
            univariate_values.append(_uni_etvd(one_t_binned[col_name],
                                               one_s_binned[col_name],
                                               col_type=col_type))

        df_univariate = pd.DataFrame({'col_names': list(original_column_type_dictionary.keys()),
                                      'uni_values': univariate_values})

        summary = summary_func(univariate_values)

    elif metric_func_name == 'bi_etvd':
        '''
        Data Prep
        '''
        sample_one_target = _sample_one_event(target_data)
        sample_one_synthetic = _sample_one_event(synthetic_data)

        one_t_binned, one_s_binned = _bin_looped(sample_one_target,
                                                 sample_one_synthetic,
                                                 original_column_type_dictionary,
                                                 number_of_bins)
        column_pairings = product(original_column_type_dictionary.keys(), original_column_type_dictionary.keys())
        '''
        Calculations
        '''

        bivariate_values = []
        column_one_pair = []
        column_two_pair = []
        for column_pairing in column_pairings:
            column_one = column_pairing[0]
            column_two = column_pairing[1]

            column_one_pair.append(column_one)
            column_two_pair.append(column_two)
            if column_one == column_two:
                # skip if column is duplicated
                bi_val = np.nan
            else:
                bi_val = _bi_etvd(one_t_binned[[column_one, column_two]],
                                  one_s_binned[[column_one, column_two]],
                                  column_one,
                                  column_two)

            bivariate_values.append(bi_val)

        df_bivariate = pd.DataFrame({'col_one': column_one_pair,
                                     'col_two': column_two_pair,
                                     'bi_vals': bivariate_values}).dropna()
        summary = summary_func(df_bivariate['bi_vals'])

    elif metric_func_name == 'correlation':
        '''
        Data Prep
        '''
        sample_two_target = _sample_two_events(target_data)
        sample_two_synthetic = _sample_two_events(synthetic_data)
        new_column_type_dictionary = _generate_column_type_dictionary(sample_two_target)

        two_t_binned, two_s_binned = _bin_looped(sample_two_target,
                                                 sample_two_synthetic,
                                                 new_column_type_dictionary,
                                                 number_of_bins)
        '''
        Calculations
        '''

        target_correlations = _calculate_correlation(two_t_binned, list(new_column_type_dictionary.keys())).fillna(0)
        syn_correlations = _calculate_correlation(two_s_binned, list(new_column_type_dictionary.keys())).fillna(0)

        correlation_differences = abs(
            target_correlations - syn_correlations.loc[target_correlations.index, target_correlations.columns])

        # dropping na values

        flattend_correlation = correlation_differences.values.flatten()
        non_null_flat_correlations = flattend_correlation[~np.isnan(flattend_correlation)]

        # perfect score = 0, if correlations exactly match in target and synthetic
        # representative of a max difference percentage
        summary = summary_func(non_null_flat_correlations)
    else:
        raise Exception("Metric function not defined")

    return summary


def _calculate_privacy_metric(target_data:DataFrame,
                              synthetic_data:DataFrame,
                              metric_func_name:str)-> float:
    """
    Compute a single metric for a given target and synthetic data set

    Please see metrics.py for a detail breakdown of each metric.

    :param target_data: target data
    :param synthetic_data: synthetic data
    :param metric_func_name: metric function to be tested
    possible options

    privacy:
    privacy_dcr_nndr - hardcoded measure

    :param summary_func: function to aggregate values (ex np.average)
    :param number_of_bins: number of bins for numeric data

    :returns: bool value if test is passed

    """
    """
    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    Data Processing

    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    """

    assert sorted(target_data.columns) == sorted(synthetic_data.columns), "Target and Synthetic have different columns"

    original_column_type_dictionary = _generate_column_type_dictionary(target_data)
    synthetic_column_type_dictionary = _generate_column_type_dictionary(synthetic_data)

    assert original_column_type_dictionary == synthetic_column_type_dictionary, "Target and Synthetic have different types"

    flat_table_target = _flatten_table(target_data.reset_index(), original_column_type_dictionary)
    flat_table_syn = _flatten_table(synthetic_data.reset_index(), original_column_type_dictionary)

    # columns now include 1st and 2nd record
    new_column_type_dictionary = _generate_column_type_dictionary(flat_table_target)


    """
    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.

    Metric Calculation

    .:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.
    """

    if metric_func_name == 'privacy_dcr_nndr':
        smoothing_factor = 1e-8
        target_data_p, syn_data_p = _prepare_data_for_privacy_metrics(flat_table_target, flat_table_syn,
                                                                      new_column_type_dictionary, smoothing_factor)
        checks,privacy_tests = _calculate_dcr_nndr(target_data_p, syn_data_p, new_column_type_dictionary, smoothing_factor)


        # two values returned, 1 means both tests passes, 0.5 means failed one test
        #average = sum(checks.values())
    else:
        raise Exception("Metric function not defined")

    return checks


# opinated truth
def compare(target_data:DataFrame,
            synthetic_data:DataFrame,
            metrics_to_return:List = ['accuracy-frequency','privacy']
            ) -> Dict:
    """

    Compare a target and synthetic dataset and returns metrics

    :param target_data: target data
    :param synthetic_data: synthetic data
    :param metrics_to_return: list of metric types to return, if empty compute all
    ['accuracy-frequency','privacy']

    :returns: univariate_total_variation_distance,bivariate_total_variation_distance,correlation_difference,distance_to_closest_record,nearest_neighbor_distance_ratio

    """
    check_common_data_format(target_data)
    check_common_data_format(synthetic_data)
    metrics_dict = {}

    if 'accuracy-frequency' in metrics_to_return:
        metrics_dict['univariate_total_variation_distance'] = _calculate_accuracy_metric(target_data, synthetic_data, 'uni_etvd')
        metrics_dict['bivariate_total_variation_distance'] = _calculate_accuracy_metric(target_data, synthetic_data, 'bi_etvd')
        metrics_dict['correlation_difference'] = _calculate_accuracy_metric(target_data, synthetic_data, 'correlation')
        #
    # percent_difference = np.mean(np.nan_to_num([uni, bi, correlation]))
    #
    # accuracy_opinion = 100 - percent_difference

    if 'privacy' in metrics_to_return:
        checks = _calculate_privacy_metric(target_data, synthetic_data, 'privacy_dcr_nndr')
        metrics_dict['distance_to_closest_record'] = checks['distance_to_closest_record']
        metrics_dict['nearest_neighbor_distance_ratio'] = checks['nearest_neighbor_distance_ratio']


    return metrics_dict