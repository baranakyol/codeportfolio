
from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, Imputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from io import StringIO
import requests


def filter_renters(data_df):
    """Select the relevant rows from the entire dataset.
    This selects the rows corresponding to renters where there is a true rent
    provided.
    Parameters
    ----------
    data_df : DataFrame
        A pandas DataFrame containing the input data
    Returns
    -------
    filtered : DataFrame
        A pandas DataFrame containing only the rows to consider
    """
    return data_df[(data_df['sc116'] == 2)      # Only renters
                   & (data_df['uf17'] < 8000)   # With a real rent provided
                   ]


def encode_column(s, field_desc):
    """Preprocess a data column.
    For binary-valued columns, replaces the value whose meaning is "yes" with
    1 and the value whose meaning is "no" with 0. Replaces all other values
    with NaN.
    For category-valued columns, replaces all valid values with a
    LabelEncoder-calculated integer. Replaces all invalid values with NaN.
    Leaves all other columns unchanged.
    Parameters
    ----------
    s : Series
        A pandas Series containing the input data
    field_desc : dict
        A dictionary with, at a minimum, a "fieldtype" entry
        specifying the column type.
    Returns
    -------
    new_series : Series
        A pandas Series containing only the rows to consider
    """
    if field_desc["fieldtype"] == "B":
        # For binary-valued columns, create a series
        # with all NaNs. Then replace all "Yes" values with 1
        # and all "No" values with 0.
        new_series = pd.Series(np.nan, index=s.index.copy())
        new_series[s == field_desc["codes"][0]] = 1
        new_series[s == field_desc["codes"][1]] = 0
        return new_series
    elif field_desc["fieldtype"] == "C":
        # For category-valued columns, first create a series
        # with all NaNs.
        valid_entries = pd.Series(np.nan, index=s.index.copy())

        # Copy all valid entries over
        valid_entries[s.isin(field_desc["categories"])] = s

        # Select the subset of valid entries
        to_code = valid_entries[s.isin(field_desc["categories"])]

        # Use LabelEncoder() to create a new ndarray with the encoded values.
        coded = LabelEncoder().fit(field_desc["categories"]).transform(to_code)

        # Put the coded values back into a pandas series with the right index
        encoded_series = pd.Series(coded, index=to_code.index.copy())

        # Finally, create te series to actually return with a combination
        # of NaNs for invalid entries and the coded values for valid rows.
        new_series = pd.Series(np.nan, index=s.index.copy())
        new_series[s.isin(field_desc["categories"])] = encoded_series
        return new_series
    else:
        return s.copy()


def preprocess_data(data_df, feature_fields, y_field):
    """Preprocess a dataset.
    Selects the relevant rows for training and testing the model
    and replaces binary- and categorical-valued columns with values
    suitable for linear models.
    This function first calls filter_renters to select only the rows
    corresponding to renters with a valid rent amount supplied.
    It then adds the categorical column 'neighborhood' corresponding to the
    combination of the boro- and sub-boro area.
    Next, it replaces all entries in the dataset corresponding to
    "Not reported", "Unknown", etc. with NaN.
    Finally, it calls encode_column to make a more convenient set of categories
    available for further use by OneHotEncoder.
    This function is run on the entire dataset, before the training/test split
    is performed, but leaks no information since it doesn't perform any
    imputation.
    Parameters
    ----------
    data_df : DataFrame
        A pandas DataFrame containing the entire dataset
    feature_fields : dict
        A dictionary whose keys are the fields to be used
        and whose values describe each individual column.
    y_fields : str
        The name of the column to be used as the y-value
    Returns
    -------
    valid_renters : DataFrame
        A pandas DataFrame containing the selected rows in their
        original form
    cleaned_data : DataFrame
        A pandas DataFrame containing the X values (features)
    rents : Series
        A pandas Series containing the y values (rents)
    """
    # Select the valid rows from the dataset
    valid_renters = filter_renters(data_df)

    # Make a new dataframe to hold the cleaned data
    cleaned_data = pd.DataFrame(index=valid_renters.index.copy())

    # Add a categorical "Neighborhood" column consisting of the boro
    # and the sub-boro
    neighborhood_series = valid_renters['boro'] * 100 + valid_renters['cd']
    copy_idx = pd.Series(neighborhood_series, neighborhood_series.index.copy())
    valid_renters = valid_renters.assign(neighborhood=copy_idx)
    feature_fields['neighborhood'] = {
        'fieldtype': 'C',
        'categories': list(set(valid_renters['neighborhood']))
    }

    # Validate each column individually according to its description
    for k, v in feature_fields.items():
        cleaned_data[k] = encode_column(valid_renters[k], v)

    return valid_renters, cleaned_data, valid_renters[y_field]


def get_data(url, seed):
    """Read, preprocess, and split the dataset.
    Parameters
    ----------
    url : str
        The URL where the dataset can be downloaded from
    seed : float
        The random seed to use for the train/test split
    Returns
    -------
    X_train : DataFrame
    X_test : DataFrame
    y_train : DataFrame
    y_test : DataFrame
    catnums : list
        The list of indices categorical columns in the feature set.
        This list can be passed to OneHotEncoder
    raw_df : DataFrame
        The entire original dataset
    """
    available_fields = {
        'boro': {'fieldtype': 'C', 'categories': range(1, 6)},
        'cd': {'fieldtype': 'C', 'categories': range(1, 19)},
        'uf1_1': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_2': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_3': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_4': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_5': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_6': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_7': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_8': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_9': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_10': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_11': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_12': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_13': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_14': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_15': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_16': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_17': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_18': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_19': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_20': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_21': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'uf1_22': {'fieldtype': 'B', 'codes': [1, 9, 8]},
        'sc24': {'fieldtype': 'B', 'codes': [1, 2, 8]},
        'sc36': {'fieldtype': 'C', 'categories': [1, 2, 3]},
        'sc37': {'fieldtype': 'C', 'categories': [1, 2, 3, 4]},
        'sc38': {'fieldtype': 'C', 'categories': [1, 2, 3]},
        'sc114': {'fieldtype': 'C', 'categories': [1, 2, 3]},
        'uf48': {'fieldtype': 'N'},
        'sc147': {'fieldtype': 'C', 'categories': [1, 2, 3]},
        'uf11': {'fieldtype': 'C', 'categories': range(1, 8)},
        'sc149': {'fieldtype': 'B', 'codes': [1, 2, None]},
        'sc173': {'fieldtype': 'C', 'categories': [1, 2, 3, 9]},
        'sc171': {'fieldtype': 'B', 'codes': [1, 2]},
        'sc150': {'fieldtype': 'N'},
        'sc151': {'fieldtype': 'N'},
        'sc154': {'fieldtype': 'C', 'categories': [1, 2, 3, 9]},
        'sc157': {'fieldtype': 'C', 'categories': [1, 2, 9]},
        'sc158': {'fieldtype': 'C', 'categories': [1, 2, 3, 4]},
        'sc185': {'fieldtype': 'B', 'codes': [0, 1, 8]},
        'sc186': {'fieldtype': 'C', 'categories': [2, 3, 4, 5, 9]},
        'sc197': {'fieldtype': 'C', 'categories': [1, 2, 3, 4]},
        'sc198': {'fieldtype': 'B', 'codes': [1, 2, 8]},
        'sc187': {'fieldtype': 'B', 'codes': [1, 2, 8]},
        'sc188': {'fieldtype': 'B', 'codes': [1, 2, 8]},
        'sc571': {'fieldtype': 'C', 'categories': range(1, 6)},
        'sc189': {'fieldtype': 'C', 'categories': range(1, 6)},
        'sc190': {'fieldtype': 'B', 'codes': [1, 2, 8]},
        'sc191': {'fieldtype': 'B', 'codes': [1, 2, 8]},
        'sc192': {'fieldtype': 'B', 'codes': [0, 1, 8]},
        'sc193': {'fieldtype': 'C', 'categories': [2, 3, 9]},
        'sc194': {'fieldtype': 'B', 'codes': [1, 2, 8]},
        'sc196': {'fieldtype': 'C', 'categories': [1, 2, 3, 4]},
        'sc199': {'fieldtype': 'C', 'categories': range(1, 6)},
        'rec15': {'fieldtype': 'C', 'categories': range(1, 14)},
        'sc26': {'fieldtype': 'C', 'categories': [12, 13, 15, 16]},
        'uf23': {'fieldtype': 'N'},
        'rec21': {'fieldtype': 'B', 'codes': [1, 2, 8]},
        'rec62': {'fieldtype': 'C', 'categories': [1, 2, 4, 5]},
        'rec64': {'fieldtype': 'C', 'categories': [1, 2, 4, 5]},
        'rec54': {'fieldtype': 'C', 'categories': range(1, 8)},
        'rec53': {'fieldtype': 'N'},
        'new_csr': {'fieldtype': 'C', 'categories': [1, 2, 5, 12, 20,
                                                     21, 22, 23, 30, 31,
                                                     80, 85, 90, 95]}
    }
    selected_fields = [
        # The borough where the apartment is located
        'boro',

        # Building type: public housing, new construction,
        # "In Rem" foreclosure, old construction
        'sc26',

        # Number of bedrooms
        'sc151',

        # Dilapidated / Not Dilapidated
        'rec21',

        # Complete plumbing facilities in unit
        'rec62',

        # Complete kitchen facilities in unit
        'rec64',

        # Maintenance deficiencies
        'rec53',

        # Building age
        'uf23',

        # Rent control/stabilization category
        'new_csr',

        # Neighborhood rating
        'sc196',

        # Wheelchair accessibility of unit
        'sc38',

        # Presence of elevator
        'sc149',

        # Building height
        'uf11',

        # Air conditioning
        'sc197',

        # Walkup
        'sc171',
    ]
    mini_fields = {k: available_fields[k]
                   for k in available_fields
                   if k in selected_fields}
    y_field = 'uf17'
    # s = requests.get(url).content
    # raw_df = pd.read_csv(StringIO(s.decode('utf-8')))
    raw_df = pd.read_csv('homework2_data.csv')
    valid_renters, validated_features, validated_rents = \
        preprocess_data(raw_df, mini_fields, y_field)
    X_train, X_test, y_train, y_test = train_test_split(
        validated_features, validated_rents, random_state=seed)
    cats = [k
            for (k, v) in mini_fields.items()
            if v["fieldtype"] == "C"]
    catnums = [i
               for (i, x) in enumerate([c in cats
                                        for c in validated_features.columns])
               if x]
    return X_train, X_test, y_train, y_test, catnums, raw_df


def model_pipeline(catnums):
    """Return the pipeline used by the model.
    Parameters
    ----------
    catnums : list
        The list of columns containing categorical features that
        should be OneHotEncoded
    Returns
    -------
    pipe : pipeline
    """
    pipe = make_pipeline(
        Imputer(strategy='most_frequent'),
        OneHotEncoder(categorical_features=catnums, sparse=False),
        PolynomialFeatures(),
        Ridge(alpha=25)
    )
    return pipe


def predict_rent(seed):
    """Return test data, true labels, and predicted labels.
    Parameters
    ----------
    seed : float
        The random state seed for the training/test split
    Returns
    -------
    test_data : ndarray
        The raw data (unpreprocessed) comprising the rows that make up the
        the holdout set. Note that per the instructors' response on Piazza to
        question @146, this is NOT the same as the X matrix that is supplied to
        ridge regression; it includes all columns, even the target variable.
    y_test : ndarray
        The true rent values
    y_pred : ndarray
        The predicted rent values
    """
    X_train, X_test, y_train, y_test, catnums, raw_df = \
        get_data("https://ndownloader.figshare.com/files/7586326", seed)
    pipe = model_pipeline(catnums)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    X_test_index = pd.DataFrame(index=X_test.index)
    return raw_df.join(X_test_index, how='inner').values, y_test.values, y_pred


def score_rent(seed):
    """Calculate the R^2 value for the model.
    Returns
    -------
    r2 : float
        The R^2 achieved by fitting the model to the training set and
        predicting the rents for the test set
    """
    X_test, y_test, y_pred = predict_rent(seed)
    return r2_score(y_test, y_pred)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, catnums, raw_df = \
        get_data("https://ndownloader.figshare.com/files/7586326", 4995)

    print(len(X_train), len(X_test), len(raw_df))
    pipe = model_pipeline(catnums)
    print("Cross-validation R2:", np.mean(
          cross_val_score(pipe, X_train, y_train)))
    print("Holdout set R2:", score_rent(4995))

    