"""
Utility functions for machine learning.
"""
from typing import Tuple, List
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, ShuffleSplit, LeaveOneGroupOut, GroupKFold, GroupShuffleSplit
import yaml


def get_hparams(hparams_file):
    """Get hyperparameter from yaml file."""
    if os.path.exists(hparams_file):
        hparams = yaml.load(open(hparams_file,"r"), Loader=yaml.FullLoader)
    else:
        raise ValueError(f'No hyperparameter file "{hparams_file}" found.')
    return(hparams)


def net_pattern(n_layers, base_size, end_size):
    """Calculates layer sizes for each layer from first and last layer so that the layer size continously increases/decreases.
    """
    if n_layers != 1:
        factor = (end_size / base_size) ** (1 / (n_layers - 1))
    else:
        factor = 1

    layer_sizes = [int(round(base_size * factor ** n)) for n in range(n_layers)]
    return (layer_sizes)


def load_data(dataset, features, targets, use_data_frac=None, shuffle=True, reset_index=True) -> pd.DataFrame:
    """
    Load ML input data from csv file.
    """
    if isinstance(targets, str):
        targets = [targets]

    data = pd.read_csv(dataset)

    # Use only a fraction of the data
    if use_data_frac is not None:
        data = data.sample(frac=use_data_frac)

    # Shuffle data
    if shuffle:
        data = data.sample(frac=1)
        if reset_index:
            data = data.reset_index(drop=True)

    # Check if dataset contains all features and targets
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f'Feature "{feature}" not found in dataset. Possible features are: {data.columns}')
    for target in targets:
        if target not in data.columns:
            raise ValueError(f'Target "{target}" not found in dataset. Possible targets are: {data.columns}')

    return data

def name_CV_col(n):
    """
    Retuns the name pattern of the CV column of `data` that contains the train and test indices for the nth cross validation run.
    """
    return f'CV_{n}'

def get_train_test_splits(df, CV, n_reps, trainfrac=None, group=None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gets train and test data doing the specified cross validation.
    """
    data_array = df.to_numpy()
    if CV == 'KFold':
        # Convert train fraction and number of total repetitions to KFold input parameters.
        if group is None:
            split = RepeatedKFold(n_splits=n_reps, n_repeats=1).split(data_array)
        else:
            groups = df[group].to_numpy()
            split = GroupKFold(n_splits=n_reps).split(data_array, groups=groups)

    elif CV == 'Random':
        assert trainfrac not in (0, 1), "ShuffleSplit won't understand that this is a fraction."
        assert trainfrac != None
        if group is None:
            split = ShuffleSplit(train_size=trainfrac, n_splits=n_reps).split(data_array)
        else:
            groups = df[group].to_numpy()
            split = GroupShuffleSplit(train_size=trainfrac, n_splits=n_reps).split(data_array, groups=groups)

    # Create a column in df for each CV split and fill it with either 'train' or 'test'.
    CV_cols = []
    for i, (train_indices, test_indices) in enumerate(split):
        n_samples = len(df)
        assert n_samples == len(train_indices) + len(test_indices)

        empty = ''
        test_or_train = pd.Series(np.full(n_samples, empty))
        test_or_train[train_indices] = 'train'
        test_or_train[test_indices] = 'test'
        # So that I can be sure that in case the indices of df and series don't align it still just adds everything in the right order.
        test_or_train = list(test_or_train)

        colname = name_CV_col(i)
        df[colname] = test_or_train
        CV_cols.append(colname)

        assert all([df[colname].iloc[idx] == 'train' for idx in train_indices])
        assert all([df[colname].iloc[idx] == 'test' for idx in test_indices])
        assert not (df[colname] == empty).any(), 'Some of the columns are neither test nor train.'

    return (df, CV_cols)