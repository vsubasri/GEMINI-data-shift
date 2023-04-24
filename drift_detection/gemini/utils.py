"""Utilities for loading and preprocessing gemini data."""
import datetime
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import importlib
import types
from cyclops.processors.impute import np_ffill_bfill, np_fill_null_num
from cyclops.processors.column_names import (
    EVENT_NAME,
    TIMESTEP,
    ADMIT_TIMESTAMP
)
from cyclops.processors.feature.vectorize import (
    split_vectorized,
    vec_index_exp,
)

from cyclops.utils.file import (
    load_pickle,
)

def get_use_case_params(dataset: str, use_case: str) -> types.ModuleType:
    """Import parameters specific to each use-case.

    Parameters
    ----------
    dataset: str
        Name of the dataset, e.g. mimiciv.
    use_case: str
        Name of the use-case, e.g. mortality_decompensation.

    Returns
    -------
    types.ModuleType
        Imported constants module with use-case parameters.

    """
    return importlib.import_module(
        ".".join(['drift_detection', dataset, use_case, "constants"])
    )

def unison_shuffled_copies(array_a, array_b):
    """Shuffle two arrays in unison."""
    assert len(array_a) == len(array_b)
    perm = np.random.permutation(len(array_a))
    return array_a[perm], array_b[perm]


def random_shuffle_and_split(x_train, y_train, x_test, y_test):
    """Randomly shuffle and split data into train and test sets."""
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    x, y = unison_shuffled_copies(x, y)

    split_index = len(x_train)
    
    x_train = x[:split_index, :]
    x_test = x[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return (x_train, y_train), (x_test, y_test)


def get_label(admin_data, X, label="mortality", encounter_id="encounter_id"):
    """Get label from admin data."""
    admin_data = admin_data.drop_duplicates(encounter_id)
    X_admin_data = admin_data[
        admin_data[encounter_id].isin(X.index.get_level_values(0))
    ]
    X_admin_data = (
        X_admin_data.set_index(encounter_id)
        .reindex(list(X.index.get_level_values(0).unique()))
        .reset_index()
    )
    y = X_admin_data[label].astype(int)
    return y

def prep(vec):
    arr = np.squeeze(vec.data, 0)
    arr = np.moveaxis(arr, 2, 1)
    arr = np.nan_to_num(arr)
    return arr

def reshape_2d_to_3d(data, num_timesteps):
    """Reshape 2D data to 3D data."""
    data = data.unstack()
    num_encounters = data.shape[0]
    data = data.values.reshape((num_encounters, num_timesteps, -1))
    return data


def flatten(X):
    """Flatten 3D data to 2D data."""
    X_flattened = X.unstack(1).dropna().to_numpy()
    return X_flattened


def temporal_mean(X):
    """Get temporal mean of data."""
    X_mean = X.groupby(level=[0]).mean()
    return X_mean


def temporal_first(X, y=None):
    """Get temporal first of data."""
    y_first = None
    X_first = X.groupby(level=[0]).first()
    if y is not None:
        y_first = y[:, 0]
    return X_first, y_first


def temporal_last(X, y):
    """Get temporal last of data."""
    X_last = X.groupby(level=[0]).last()
    num_timesteps = y.shape[1]
    y_last = y[:, (num_timesteps - 1)]
    return X_last, y_last


def get_numerical_cols(X: pd.DataFrame):
    """Get numerical columns of temporal dataframe."""
    numerical_cols = [
        col for col in X if not np.isin(X[col].dropna().unique(), [0, 1]).all()
    ]
    return numerical_cols


def scale(X: pd.DataFrame):
    """Scale columns of temporal dataframe.

    Returns
    -------
    model: torch.nn.Module
        feed forward neural network model.

    """
    numerical_cols = get_numerical_cols(X)
    for col in numerical_cols:
        scaler = StandardScaler().fit(X[col].values.reshape(-1, 1))
        X[col] = pd.Series(
            np.squeeze(scaler.transform(X[col].values.reshape(-1, 1))),
            index=X[col].index,
        )
    return X


def reformat(X, aggregation_type):
    """Normalize data."""
    y = None

    if aggregation_type == "mean":
        X_normalized = temporal_mean(X)
    elif aggregation_type == "first":
        (
            X_normalized,
            y,
        ) = temporal_first(X, y)
    elif aggregation_type == "last":
        (
            X_normalized,
            y,
        ) = temporal_last(X, y)
    elif aggregation_type == "time_flatten":
        X_normalized = X.copy()
    elif aggregation_type == "time":
        X_normalized = X.copy()
    else:
        raise ValueError("Incorrect Aggregation Type")
    return X_normalized


def process(X, aggregation_type, timesteps):
    """Process data."""
    if aggregation_type == "time_flatten":
        X_preprocessed = flatten(X)
    elif aggregation_type == "time":
        X_preprocessed = reshape_2d_to_3d(X, timesteps)
    else:
        X_preprocessed = X.dropna().to_numpy()
    return X_preprocessed

def impute(temp_vec):
    # Forward fill then backward fill to get rid of all of the timestep nulls
    temp_vec.impute_over_axis(TIMESTEP, np_ffill_bfill)

    # Fill those all-null timesteps with feature mean
    # (since forward and backward filling still leaves them all null)
    axis = temp_vec.get_axis(EVENT_NAME)

    for i in range(temp_vec.data.shape[axis]):
        index_exp = vec_index_exp[:, :, i]
        data_slice = temp_vec.data[index_exp]
        mean = np.nanmean(data_slice)
        func = lambda x: np_fill_null_num(x, mean)  # noqa: E731
        temp_vec.impute_over_axis(TIMESTEP, func, index_exp=index_exp)

    return temp_vec

def compute_timestep(timestamps, timestep_size, event):
    timestamps[f"{event}_after_admit"] = timestamps[event] - timestamps[ADMIT_TIMESTAMP]
    timestamps[f"{event}_timestep"] = (
        timestamps[f"{event}_after_admit"]
        / pd.Timedelta(f"{timestep_size} hour")
    ).apply(np.floor)
    return timestamps

def get_source_target(tab_features, tab_vectorized, dataset, splice_map, train_frac=0.8, axis="encounter_id"):
    """Get dataset for hospital."""
    
    if dataset == "simulated_deployment":
        ids_source = tab_features.slice(splice_map, "admit_timestamp > '2011-04-01' & admit_timestamp < '2019-01-01'")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "admit_timestamp > '2019-01-01' & admit_timestamp < '2020-08-01'")
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "covid":
        ids_source = tab_features.slice(splice_map, "admit_timestamp > '2019-01-01' & admit_timestamp < '2020-02-01'")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "admit_timestamp > '2020-03-01' & admit_timestamp < '2020-08-01'")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "hosp_type_community":
        splice_map['hospital_id'] = ["THPC", "THPM"]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['hospital_id'] = ["SMH", "MSH", "UHNTG", "UHNTW", "SBK"]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "hosp_type_academic":
        splice_map['hospital_id'] = ["SMH", "MSH", "UHNTG", "UHNTW", "SBK"]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['hospital_id'] = ["THPC", "THPM"]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "night":
        ids_source = tab_features.slice(splice_map, "night_shift == 1")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "night_shift == 0")
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "day":
        ids_source = tab_features.slice(splice_map, "night_shift == 0")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "night_shift == 1")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "female":
        ids_source = tab_features.slice(splice_map, "sex == 0")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "sex == 1")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "male":
        ids_source = tab_features.slice(splice_map, "sex == 1")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "sex == 0")
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "pediatric":
        ids_source = tab_features.slice(splice_map, "age >= 18")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "age < 18")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "adult_18_29":
        ids_source = tab_features.slice(splice_map, "age > 18 & age < 29")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "age > 29 | age < 18")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "adult_30_44":
        ids_source = tab_features.slice(splice_map, "age > 30 & age < 44")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "age > 44 | age < 30")
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "adult_45_64":
        ids_source = tab_features.slice(splice_map, "age > 45 & age < 64")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "age > 64 | age < 45")
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "geriatric":
        ids_source = tab_features.slice(splice_map, "age > 65")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "age <= 65")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "nursing_home":
        ids_source = tab_features.slice(splice_map, "from_nursing_home_mapped == 1")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "from_nursing_home_mapped == 0")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "not_nursing_home":
        ids_source = tab_features.slice(splice_map, "from_nursing_home_mapped == 0")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "from_nursing_home_mapped == 1")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "acute_care":
        ids_source = tab_features.slice(splice_map, "from_acute_care_institution_mapped == 1")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "from_acute_care_institution_mapped == 0")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "not_acute_care":
        ids_source = tab_features.slice(splice_map, "from_acute_care_institution_mapped == 0")
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "from_acute_care_institution_mapped == 1")
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "seasonal_summer":
        splice_map['admit_month'] = [6, 7, 8, 9]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['admit_month'] = [11, 12, 1, 2]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "seasonal_winter":
        splice_map['admit_month'] = [11, 12, 1, 2]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['admit_month'] = [6, 7, 8, 9]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "weekday":
        splice_map['admit_day'] = [0, 1, 2, 3, 4]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['admit_day'] = [5, 6]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "weekend":
        splice_map['admit_day'] = [5, 6]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['admit_day'] = [0, 1, 2, 3, 4]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "weekend_baseline":
        splice_map['admit_day'] = [5, 6]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['admit_day'] = [5, 6]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "weekday_baseline":
        splice_map['admit_day'] = [0, 1, 2, 3, 4]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['admit_day'] = [0, 1, 2, 3, 4]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "covid_baseline":
        dataset_ids = tab_features.slice(splice_map, "admit_timestamp > '2019-01-01' & admit_timestamp < '2020-02-01'")
        x = tab_vectorized.take_with_index(axis, dataset_ids)
        x_s, x_t = split_vectorized(
            [x],
            [train_frac, 1-train_frac],
            axes=axis,
        )[0]

    elif dataset == "male_baseline":
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "sex == 1")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "female_baseline":
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "sex == 0")
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "day_baseline":
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "night_shift == 0")
        x_t = tab_vectorized.take_with_index(axis, ids_target)
        
    elif dataset == "night_baseline":
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        ids_target = tab_features.slice(splice_map, "night_shift == 1")
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "seasonal_summer_baseline":
        splice_map['admit_month'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['admit_month'] = [6, 7, 8, 9]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "seasonal_winter_baseline":
        splice_map['admit_month'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ids_source = tab_features.slice(splice_map)
        x_s = tab_vectorized.take_with_index(axis, ids_source)
        splice_map['admit_month'] = [11, 12, 1, 2]
        ids_target = tab_features.slice(splice_map)
        x_t = tab_vectorized.take_with_index(axis, ids_target)

    elif dataset == "hosp_type_academic_baseline":
        splice_map['hospital_id'] = ["SMH", "MSH", "UHNTG", "UHNTW", "PMH","SBK"]
        dataset_ids = tab_features.slice(splice_map)
        x = tab_vectorized.take_with_index(axis, dataset_ids)
        x_s, x_t = split_vectorized(
            [x],
            [train_frac, 1-train_frac],
            axes=axis,
        )[0]
    elif dataset == "hosp_type_community_baseline":
        splice_map['hospital_id'] = ["THPC", "THPM"]
        dataset_ids = tab_features.slice(splice_map)
        x = tab_vectorized.take_with_index(axis, dataset_ids)
        x_s, x_t = split_vectorized(
            [x],
            [train_frac, 1-train_frac],
            axes=axis,
        )[0]
    elif dataset == "random":
        dataset_ids = tab_features.slice(splice_map)
        x = tab_vectorized.take_with_index(axis, dataset_ids)
        x_s, x_t = split_vectorized(
            [x],
            [train_frac, 1-train_frac],
            axes=axis,
        )[0]

    return (x_s, x_t)

def import_dataset_hospital(
    tab_vec_comb, data_split, shuffle=True,
):

    X_train_vec = load_pickle(tab_vec_comb + "comb_train_X_"+data_split)
    y_train_vec = load_pickle(tab_vec_comb + "comb_train_y_"+data_split)
    X_val_vec = load_pickle(tab_vec_comb + "comb_val_X_"+data_split)
    y_val_vec = load_pickle(tab_vec_comb + "comb_val_y_"+data_split)
    X_test_vec = load_pickle(tab_vec_comb + "comb_test_X_"+data_split)
    y_test_vec = load_pickle(tab_vec_comb + "comb_test_y_"+data_split)

    X_train = prep(X_train_vec.data)
    y_train = prep(y_train_vec.data)
    X_val = prep(X_val_vec.data)
    y_val = prep(y_val_vec.data)
    X_test = prep(X_test_vec.data)
    y_test = prep(y_test_vec.data)
    
    if shuffle:
        random_shuffle_and_split(X_train, y_train, X_val, y_val)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)