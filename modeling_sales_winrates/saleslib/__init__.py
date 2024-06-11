#!/opt/homebrew/anacoda3/bin/python
# Tim H 2024
# __init__.py
"""Python module for processing datasets related to CRM sales"""

# Caution: normalizing and standardizing using relative values will change
# between train and validation datasets

# TODO: create a single function that calculates and summarizes the changes
#       that a single pipeline function performed.

# Define some exclusions for PEP8 that don't apply when the Jupyter Notebook
#   is exported to .py file
# pylint: disable=pointless-statement
# pylint: disable=fixme
# pylint: disable=expression-not-assigned
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
# pylint: disable=import-error
# pylint: disable=line-too-long

__version__ = '0.3.3'

import json
import time
# import os
# import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, recall_score, auc, average_precision_score, balanced_accuracy_score, f1_score, log_loss, precision_score
from xgboost import XGBClassifier

###############################################################################
#   Defining constants
#   TODO: migrate to separate config file
###############################################################################
RANDOM_STATE = 1234

LABEL_COLUMN_NAME = "Won"

WINSORIZING_RANGE_VALUE_MIN = {
    'Age of opp in days': 30,
    'Deal size (USD)': 10000,
    'Num times opp pushed': 0,   # must have all columns that will be normalized
    'quarter_created':      1,
    'quarter_closed':       1,
}

WINSORIZING_RANGE_VALUE_MAX = {
    'Age of opp in days': 365,
    'Deal size (USD)': 180000,
    'Num times opp pushed': 6,
    'quarter_created':      4,   # must have all columns that will be normalized
    'quarter_closed':       4,
}

COLUMNS_TO_NORMALIZE_TO_FIXED_RANGE = [
    'Age of opp in days',
    'Deal size (USD)',
    'Num times opp pushed',
    'quarter_created',
    'quarter_closed'
    ]

CONVERT_BOOLEAN_TO = 'uint8[pyarrow]'   # for XGBoost

# save for last since other functions use the original names
COLUMN_NAME_MAPPINGS = {
    #'CRM Identifier':       'opp_id',      # column dropped earlier
    #'Customer Sector':      'industry',
    'Sales Rep ID':         'sales_rep',
    'Sales team name':      'sales_team',
    #'primary_product':      'primary_product',
    'Num times opp pushed': 'pushes',
    'quarter_created':      'quarter_created',
    'quarter_closed':       'quarter_closed',
    'Age of opp in days':   'age',
    LABEL_COLUMN_NAME:                  LABEL_COLUMN_NAME,
    'partner_involved':     'partner',
    'Deal size (USD)':      'revenue'
}

UNUSED_COLUMNS_TO_DROP = ['CRM Identifier']

COLUMN_DROP_ROWS_BELOW_VALUES={
    'quarter_created':      1,
    'quarter_closed':       1,
    'Age of opp in days':   14,
    'Deal size (USD)':      1500
}

COLUMN_DROP_ROWS_ABOVE_VALUES={
    'quarter_created':      4,
    'quarter_closed':       4,
    'Age of opp in days':   720,
    'Deal size (USD)':      1000000,
}

COLUMN_DATATYPES_MAPPINGS = {
    #'CRM Identifier':       'string[pyarrow]', # column dropped earlier
    # 'Customer Sector':      'string[pyarrow]',
    'Industry':             'string[pyarrow]',
    'Sales Rep ID':         'string[pyarrow]',
    'Sales team name':      'string[pyarrow]',
    'primary_product':      'string[pyarrow]',
    'Num times opp pushed': 'uint8[pyarrow]',
    'quarter_created':      'uint8[pyarrow]',
    'quarter_closed':       'uint8[pyarrow]',
    'Age of opp in days':   'uint16[pyarrow]',
    LABEL_COLUMN_NAME:      'bool[pyarrow]',
    'partner_involved':     'bool[pyarrow]',
    'Deal size (USD)':      'uint32'
}

COLUMNS_TO_OHE = ['Industry', 'Sales Rep ID', 'Sales team name',
                  #'primary_product',
                  'positioning_category',
                  'product_family', 'hosting_location', 'sales_territory']

MANDATORY_NONEMPTY_COLUMNS = [
                              'Sales Rep ID',
                              #'CRM Identifier',        # column dropped earlier
                              'Sales team name',
                              'Age of opp in days',
                              LABEL_COLUMN_NAME,
                              'Deal size (USD)']

# https://matplotlib.org/stable/users/explain/colors/colormaps.html
FULL_COLORS_LIST = ["tab:blue",   "tab:orange", "tab:green", "tab:red",
                        "tab:purple", "tab:pink",   "tab:brown", "tab:cyan",
                        "tab: gray",  "tab: olive"]


###############################################################################
#   Defining methods
###############################################################################


def train_and_evaluate_optimal_model_from_gridsearch(grid_search_obj, metric_name, X_train, y_train):
    """Method for evaluating and displaying various metrics about the
    optimal model that was determined from a Grid Search
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    TODO: list the secondary metrics for the optimal model
    """

    #print(f'GridSearchCV results for estimator type: {type(grid_search_obj.best_estimator_).__name__}')    # does not work if scoring is a string instead of dict
    print(f'* Optimizing for metric: {metric_name}')
    print(f'* CV value (# of folds): {grid_search_obj.n_splits_}')
    print(f'* Grid Search through: {grid_search_obj.param_grid}')
    print(f'* Best {metric_name} score found: {grid_search_obj.best_score_:0.4f}')
    print(f'* Parameters associated with best {metric_name}:    {grid_search_obj.best_params_}')
    bst = XGBClassifier(**grid_search_obj.best_params_)

    model_train_time_start = time.time()
    bst.fit(X_train, y_train)
    model_train_time_stop = time.time()
    model_train_time_total = model_train_time_stop - model_train_time_start

    #.cv_results_ does not offer training times on specific models or on the grid
    # search as a whole.
    print(f'* Optimal model train time: {model_train_time_total:.1f} sec')
    return bst


def set_datatypes(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method for setting datatypes on a dataframe using a dict"""
    df_opps = df_opps.astype(COLUMN_DATATYPES_MAPPINGS)
    df_opps[LABEL_COLUMN_NAME]  = df_opps[LABEL_COLUMN_NAME].astype('uint8[pyarrow]')
    df_opps['partner_involved'] = df_opps['partner_involved'].astype('uint8[pyarrow]')
    print('[Set Datatypes] Recast datatypes')
    return df_opps


def drop_rows_outside_ranges(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method that removes rows outside
    (not inclusive) of the valid ranges. Designed to remove Opps that were way
    too old, or only created for quoting, and aren't indicitive of
    larger patterns"""
    # using separate loops since a given key may or may not have both an
    # upper and lower limit
    rows_before = df_opps.shape[0]
    for colname, lower_threshold  in COLUMN_DROP_ROWS_BELOW_VALUES.items():
        indexes_to_drop = df_opps[df_opps[colname] < lower_threshold].index
        num_to_drop = len(indexes_to_drop)
        if num_to_drop > 0:
            print(f'[Drop Rows outside range] Dropping {num_to_drop} rows ({num_to_drop/rows_before*100:.2f}%) from {colname} because value was < {lower_threshold}')
        df_opps.drop(indexes_to_drop, inplace=True)

    for colname, upper_threshold in COLUMN_DROP_ROWS_ABOVE_VALUES.items():
        indexes_to_drop = df_opps[df_opps[colname] > upper_threshold].index
        num_to_drop = len(indexes_to_drop)
        if num_to_drop > 0:
            print(f'[Drop Rows outside range] Dropping {num_to_drop} rows ({num_to_drop/rows_before*100:.2f}%) from {colname} because value was > {upper_threshold}')
        df_opps.drop(indexes_to_drop, inplace=True)

    rows_after = df_opps.shape[0]
    total_dropped = rows_before - rows_after
    if total_dropped > 0:
        print(f'[Drop Rows outside range] Dropped a total of {total_dropped} rows ({100.0*total_dropped/rows_before:.2f}%) because values were outside ranges')
    return df_opps


def drop_rows_missing_values_in_columns(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method for dropping rows that have empty (None/NaN) values in
    a list of specified columns. Only enforces on column names listed in
    MANDATORY_NONEMPTY_COLUMNS
    """
    rows_before = df_opps.shape[0]
    df_opps.dropna(subset=MANDATORY_NONEMPTY_COLUMNS, inplace=True)
    rows_after = df_opps.shape[0]
    total_dropped = rows_before - rows_after
    if total_dropped > 0:
        print(f'[Drop Rows Missing Values] Dropped a total of {total_dropped} rows ({100.0*total_dropped/rows_before:.2f}%) because values were missing when they were mandatory')
    return df_opps


def winsorize_low_end_by_value(df_opps: pd.DataFrame, colname: str,
                               low_end_value) -> pd.DataFrame:
    """Pipeline method - forces all values below a specified value threshold
    to be that threshold. Does not use percentages.
    """
    num_to_winsorize = len(df_opps[df_opps[colname] < low_end_value].index)
    if num_to_winsorize > 0:
        df_opps[colname] = df_opps[colname].clip(lower=low_end_value)
        print(f'[Winsorized Low End] Winsorized {num_to_winsorize} rows ({100.0*num_to_winsorize/df_opps.shape[0]:.2f}%) in feature \"{colname}\" because value was < {low_end_value}')
    # add assert that all values are in range?
    return df_opps


def winsorize_high_end_by_value(df_opps: pd.DataFrame, colname: str,
                                high_end_value) -> pd.DataFrame:
    """Pipeline method - forces all values above a specified value threshold
    to be that threshold. Does not use percentages.
    """
    num_to_winsorize = len(df_opps[df_opps[colname] > high_end_value].index)
    if num_to_winsorize > 0:
        df_opps[colname] = df_opps[colname].clip(upper=high_end_value)
        print(f'[Winsorized High End] Winsorized {num_to_winsorize} rows ({100.0*num_to_winsorize/df_opps.shape[0]:.2f}%) in feature \"{colname}\" because value was > {high_end_value}')
    # add assert that all values are in range?
    return df_opps


def winsorize_cols(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method for clipping upper and lower ends on a series of columns.
    Use this instead of winsorize_high_end_by_value or
    winsorize_low_end_by_value.
    Does not use percentages.
    """
    dfResult = df_opps.copy()

    for colname, lower_threshold in WINSORIZING_RANGE_VALUE_MIN.items():
        dfResult = winsorize_low_end_by_value(dfResult, colname, lower_threshold)

    for colname, upper_threshold in WINSORIZING_RANGE_VALUE_MAX.items():
        dfResult = winsorize_high_end_by_value(dfResult, colname, upper_threshold)
    return dfResult


def normalize_relative_to_0_1(df_opps, column_name) -> pd.DataFrame:
    """Pipeline method for normalizing (scaling) values in a column to [0,1]. Takes the
    minimum in the column and forces that to be 0. The maximum becomes 1. Does
    not adjust means or standard deviations.
    """
    starting_min = df_opps[column_name].min()
    starting_max = df_opps[column_name].max()
    df_opps[column_name] = (df_opps[column_name] - starting_min) / (starting_max - starting_min)
    ending_min = df_opps[column_name].min()
    ending_max = df_opps[column_name].max()
    print(f'[Normalized 0-1] Normalized range of feature \"{column_name}\" from [{starting_min},{starting_max}] to [{ending_min},{ending_max}]')
    return df_opps


def normalize_to_absolute_range_then_01(df_opps, column_name) -> pd.DataFrame:
    """ Pipeline method for normalizing a single column's data to a specific
    range.
    For example:
        1) df_opps's 'Deal size (USD)' column has values ranging from [15000, 80000]
        2) Based on WINSORIZING_RANGE_VALUE_MIN and WINSORIZING_RANGE_VALUE_MAX,
            the absolute range is [10000, 180000]
        3) The column is rescaled so that 10000 is 0 and 180000 is 1
    Necessary because a list of opps may not run the full range, and performing
    a relative scale (normalizing) will result in inconsistent values when
    predicting.
    """
    # gather the min & max for values in the current dataframe
    starting_min = df_opps[column_name].min()
    starting_max = df_opps[column_name].max()

    # gather the CONSTANTS defined for this specific column. These are the
    # absolute values that are used to define [0,1] after normalization.
    NORMALIZE_LOWER_END = WINSORIZING_RANGE_VALUE_MIN[column_name]  # will be normalized to 0
    NORMALIZE_UPPER_END = WINSORIZING_RANGE_VALUE_MAX[column_name]  # will be normalized to 1

    # Normalize it to the absolute range
    df_opps[column_name] = (df_opps[column_name] - NORMALIZE_LOWER_END) / (NORMALIZE_UPPER_END - NORMALIZE_LOWER_END)

    # get the new range for display
    ending_min = df_opps[column_name].min()
    ending_max = df_opps[column_name].max()
    print(f'[Normalized Absolute] Normalized range of feature \"{column_name}\" from [{starting_min},{starting_max}] to [{ending_min},{ending_max}] using absolute range of [{NORMALIZE_LOWER_END},{NORMALIZE_UPPER_END}]')

    return df_opps


def normalize_cols_to_fixed_range(df_opps) -> pd.DataFrame:
    """Pipeline method for scaling values in a set of columns to a
    relative range.
    """
    # loop through a defined list of columns
    for colname in COLUMNS_TO_NORMALIZE_TO_FIXED_RANGE:
        df_opps = normalize_to_absolute_range_then_01(df_opps, colname)
    return df_opps


def convert_all_boolean_cols_to_int(df_opps) -> pd.DataFrame:
    """Pipeline method for converting all Boolean features into numeric
    datatypes. I think XGBoost requires this.
    """
    num_cols_converted = 0

    # iterate through ALL columns
    for colname in df_opps.columns:
        # there are multiple datatypes that are boolean, depending on which
        # engine is being used, it could be 'bool', 'boolean', 'bool[pyarrow]'
        # or others

        if "bool" in str(df_opps[colname].dtype):
            num_cols_converted += 1
            df_opps[colname] = df_opps[colname].astype(CONVERT_BOOLEAN_TO)

    if num_cols_converted > 0:
        print(f'[Convert] Converted {num_cols_converted} columns from Boolean to {CONVERT_BOOLEAN_TO} for XGBoost')

    return df_opps


def convert_json_to_features(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Converts strings that contain JSON into separate columns. This was
    necessary because Salesforce configuratoins may enforce a limit
    on the number of columns that can be exported in a report. This is a way
    to export more data than a CRM provides for.

    TODO: vectorize this whole function instead of iterating through rows,
            probably by writing a new function and using .apply
    TODO: explicitly define datatypes for new columns
    """

    # Establishing invalid default values to create the features
    df_opps['quarter_created']  = 0
    df_opps['quarter_closed']   = 0
    df_opps['primary_product']  = None
    df_opps['partner_involved'] = None

    # convert JSON to new columns
    # iterate through each row - I know this isn't great.
    for index_iter in df_opps.index:
        fields_as_json_str = df_opps['array_of_sfdc_formulas'][index_iter]
        fields_as_dict = json.loads(fields_as_json_str)
        for colname, value in fields_as_dict.items():
            df_opps.loc[df_opps.index == index_iter, colname] = value

    df_opps.drop(columns='array_of_sfdc_formulas', inplace=True)
    return df_opps


def rename_features(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method for renaming columns in a dataframe.
    This should be the very last function in the pipeline
    TODO: low priority - give more detail about which or how many columns were
            changed by this method. Avoid displaying anything if nothing changed.
    """
    df_opps.rename(columns=COLUMN_NAME_MAPPINGS, inplace=True)
    print('[Rename Features] Renamed column names')
    return df_opps


def standardize_column_using_zscore(df_opps, column_name) -> pd.DataFrame:
    """Pipeline method for standardizing a column's values using Z-Score
    methods. The new mean will be 0 and the new standard deviation will be 1.
    """
    print(f'[Standardize Column (Z-Score)] Standardized feature \"{column_name}\"')
    df_opps[column_name] = (df_opps[column_name] - df_opps[column_name].mean()) / df_opps[column_name].std()
    return df_opps


def onehotencode_single_column(df_opps: pd.DataFrame,
                               colname: str) -> pd.DataFrame:
    """One-hot encodes a single column to many columns in a data frame.
    Removes the original column. Sets the OHE columns' prefix to be the original
    column's name.
    Example:
        1) df_opp's column named 'Sales Rep ID' contains two unique values:
            John Doe and Jane Smith
        2) Two new columns are created named 'Sales Rep ID_John Doe' and
            'Sales Rep ID_Jane Smith'
        3) The original column 'Sales Rep ID' is dropped.

    TODO: could be switched to a single call for multiple columns

    https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
    """
    # create a one-hot encoded version in a new dataframe
    temp_df = pd.get_dummies(df_opps[colname], prefix=colname)

    # merge the new dataframe into the existing one
    df_opps = df_opps.join(temp_df)
    num_new_features = temp_df.shape[1]

    # remove the original column now that it has been encoded
    # into the existing dataframe
    df_opps.drop(columns=colname, inplace=True)

    print(f'[One Hot Encode] Encoded feature \"{colname}\", added {num_new_features} new columns. Dropped original.')
    return df_opps


def drop_unused_columns(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method to drop columns listed in UNUSED_COLUMNS_TO_DROP"""
    num_columns_before = df_opps.shape[1]
    df_opps.drop(columns=UNUSED_COLUMNS_TO_DROP, inplace=True)
    num_columns_after = df_opps.shape[1]
    print(f'[Drop Unused Columns] Dropped {num_columns_after - num_columns_before} columns')
    return df_opps


def onehotencode_string_columns(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method to One Hot Encode each column listed in COLUMNS_TO_OHE"""
    for colname in COLUMNS_TO_OHE:
        df_opps = onehotencode_single_column(df_opps, colname)
    return df_opps


def add_feature_sales_team(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method to add a new column (feature) that extracts the sales
    team's larger territory. There are often patterns within larger territories
    beyond the immediate sales team.
    Extracts the string up to, but not including the first space.
    For example:
        If 'Sales team name' is 'Central - North',
        then 'sales_territory' is 'Central'
    """
    df_opps['sales_territory'] = df_opps['Sales team name'].str.extract(r'([^\s]+)',
                                     expand = True)
    return df_opps


def add_feature_is_managed_service(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method to add a new column (feature) that determines if the
    primary solution being positioned is a Managed Service. Relies on specific
    formatting of the primary_product column.
    """
    df_opps['is_managed_service'] = df_opps['primary_product'].str.contains('managed_service')
    return df_opps


def add_features_sales_type(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method to add 3 new columns (features) that extract information
    about the primary_product being positioned. The 'primary_product' column
    is assumed to be in the pipe delimited following format:
        'Positioning_category|product_family|hosting_location'
    Positioning Category: is the solution a software product or a Managed Service?
    Product Family:   The name of the product line (which may have a managed service, or on-prem version)
    Hosting Location: is the product on-prem or SaaS? Irrelevant for Managed Services

    For example: a 'primary_product' has the value 'product|C|saas_platform'
        positioning_category = 'product'
        product_family       = 'C'
        hosting_location     = 'saas_platform'

    TODO: vectorize using .apply(). This is inefficent.
    """

    # SALE_TYPE_CATEGORIES = ['product', 'managed_service', 'unclear', 'limited_service_engagement']

    # initialize new features/columns
    df_opps['positioning_category'] = None # pd.Categorical(SALE_TYPE_CATEGORIES)
    df_opps['product_family']       = None
    df_opps['hosting_location']     = None


    # convert | delimited to new columns
    # iterates through each row
    for index_iter in df_opps.index:
        fields_as_pipe_delim_str = df_opps['primary_product'][index_iter]
        if fields_as_pipe_delim_str is None or fields_as_pipe_delim_str == '':
            df_opps.loc[df_opps.index == index_iter, 'positioning_category'] = None
            df_opps.loc[df_opps.index == index_iter, 'product_family']       = None
            df_opps.loc[df_opps.index == index_iter, 'positioning_category'] = None
            continue    # early exit
        fields_as_list = fields_as_pipe_delim_str.split('|')

        cat = fields_as_list[0]
        if cat is None or cat == '':
            df_opps.loc[df_opps.index == index_iter, 'positioning_category'] = None
        else:
            df_opps.loc[df_opps.index == index_iter, 'positioning_category'] = cat

        fam = fields_as_list[1]
        if fam is None or fam == '':
            df_opps.loc[df_opps.index == index_iter, 'product_family'] = None
        else:
            df_opps.loc[df_opps.index == index_iter, 'product_family'] = fam

        host = fields_as_list[2]
        if host is None or fam == '':
            df_opps.loc[df_opps.index == index_iter, 'hosting_location'] = None
        else:
            df_opps.loc[df_opps.index == index_iter, 'hosting_location'] = host

    df_opps.drop(columns='primary_product', inplace=True)
    return df_opps


def calc_win_loss_data(df_opps: pd.DataFrame):
    """Calculates the win/loss ratio of a dataframe of opps.
    Useful for rebalancing (upscaling/downscaling) datasets and
    creating representative samples
    """
    opps_lost = df_opps[df_opps[LABEL_COLUMN_NAME] == 0]
    num_losses = opps_lost.shape[0]
    opps_won  = df_opps[df_opps[LABEL_COLUMN_NAME] == 1]
    num_wins = opps_won.shape[0]
    total_opps = df_opps.shape[0]
    win_ratio = num_wins / total_opps
    print(f'[WinLossInfo] Wins: {num_wins} {win_ratio*100.0:.1f}%  Losses: {num_losses}  Total: {total_opps}')
    return opps_won, opps_lost, win_ratio, num_losses, num_wins, total_opps


def upsample_wins(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method that returns a new dataframe of Opps that have all
    wins upsampled by 60% won opps are repeated, not generated or changed.
    Useful for data pipelining
    TODO: implement a parameter that sets the desired percentage of wins AFTER
          the upsampling
    """
    # desired_win_ratio = 0.333

    #print(f'[Upsampling Wins] Attempting to get win ratio to {desired_win_ratio}')
    opps_won, opps_lost, win_ratio, num_losses, num_wins, total_opps = calc_win_loss_data(df_opps)

    # upsampled_win_count = int(num_wins * (desired_win_ratio / win_ratio))
    upsampled_win_count = int(num_wins * 1.6)

    print(f'[Upsampling Wins] Resampling wins from {num_wins} opps to {upsampled_win_count} opps...')
    df_win_upsampled = resample(opps_won,
                replace=True,
                n_samples=upsampled_win_count,
                random_state=RANDOM_STATE)

    data_upsampled = pd.concat([df_win_upsampled, opps_lost])
    calc_win_loss_data(data_upsampled)
    return data_upsampled


def get_color_list(scoring_list):
    """Returns a list of color names to be used in a plot's legend.
    Based on the number of scoring entries in a dict.
    Designed to be used by plot_grid_search_scores.
    https://matplotlib.org/stable/gallery/color/named_colors.html
    """

    num_entries = len(scoring_list)
    max_num_colors = len(FULL_COLORS_LIST)
    if num_entries > max_num_colors:
        raise IndexError("more colors requested than available")
    return FULL_COLORS_LIST[0:num_entries]


def plot_grid_search_scores(param_to_study, clf, scoring, scale='linear'):
    """Creates a plot of scoring metric across a single hyperparameter from a
    Grid Search.
    For example: After a GridSearchCV run, hand off the results to this
    function to plot how recall (or any other metric in 'scoring') varies
    with respect to a single hyperparameter (such as tree_depth)."""

    PLOT_SIZE  = 10                                 # assuming a square plot
    results    = clf.cv_results_
    model_name = type(clf.best_estimator_).__name__ # pull the Estimator's name
    cv         = clf.n_splits_                      # get the number of folds

    num_samples_for_param = len(clf.param_grid[param_to_study])
    # don't bother ploting if there's only 1 value for this hyperparameter
    if num_samples_for_param < 2:
        print(f'Will not plot {param_to_study} since it only has {num_samples_for_param} data point')
        return

    plt.figure(figsize=(PLOT_SIZE, PLOT_SIZE))
    plt.title(f"{model_name} model performance vs hyperparameter {param_to_study}\nCV={cv}", fontsize=14)

    colors = get_color_list(scoring)

    plt.xlabel(param_to_study)
    plt.ylabel("Score")

    ax = plt.gca()
    #ax.set_xlim(0, 402)
    #ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results["param_" + param_to_study].data, dtype=float)

    for scorer, color in zip(sorted(scoring), colors):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
            sample_score_std = results["std_%s_%s" % (sample, scorer)]
            ax.fill_between(
                X_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1 if sample == "test" else 0,
                color=color,
            )
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == "test" else 0.7,
                label="%s (%s)" % (scorer, sample),
            )

        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot(
            [
                X_axis[best_index],
            ]
            * 2,
            [0, best_score],
            linestyle="-.",
            color=color,
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    # put the legend off to the right side so it doesn't block the graph
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xscale(scale)
    plt.grid(False)
    plt.show()

    return ax  # return the plot so I can modify the plot outside this function


# duplicates functionality of assert_datatypes_ready_for_training
def verify_data_ready_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method for validating that all values in a dataframe
    are suitable for training.
    Throws an Exception if any strings or other types are detected.
    This is useful for a last minute check in a pipeline before training or
    validating. Returns the dataframe if successful.
    """
    # verify that there are only numeric and boolean datatypes left
    # there should not be any strings left
    for index, value in df.dtypes.items():
        assert value in [
            "float64",
            "bool",
            "int64",
        ], f"Column name {index} is not numeric or boolean- found {value}. All features at this point should be numeric or boolean. Exiting."

    print("Feature datatype check passed.")

    # get a Panda Series of the columns and number of NaNs in each one
    nan_count = np.sum(df.isnull(), axis=0)

    # iterate through the Series. It could be easier to just throw and exception if
    # any have a value of zero.
    for index, value in nan_count.items():
        assert (
            value == 0
        ), f"Column name {df.columns[index]} (index = {index}) has {value} missing values (NaN). Model cannot have any missing values. Exiting."

    return df


def assert_datatypes_ready_for_training(df_opps: pd.DataFrame) -> pd.DataFrame:
    """Pipeline method to validate that all columns are numeric.
    https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html

    TODO: provide more info when a column fails
    """

    all_dtypes_are_numeric = True
    for colname, dt in df_opps.dtypes.to_dict().items():
        if not pd.api.types.is_numeric_dtype(dt):
            all_dtypes_are_numeric = False
            print(f'[Datatype Check] Column \"{colname}\" is not numeric, it is {dt}')

    assert all_dtypes_are_numeric, 'One or more features are not numeric. Exiting.'
    return df_opps


def list_unique_string_values(df_opps: pd.DataFrame, MAX_UNIQUE_COLUMN_VALUES=20) -> dict:
    """Exploratory Data Anlaysis
    Returns a dict listing column names that have a small number of unique values
    and those corresponding values. Good for investigating which columns
    to One Hot Encode."""

    # get a Pandas Series that describes ALL of the column names and # of
    # unique values in each column/feature
    unique_column_names = df_opps.describe(include='all').loc['unique']

    # get the top MAX_UNIQUE_COLUMN_VALUES of features that have the least
    # prevalent unique values, store in a List
    column_names_with_few_unique_values = unique_column_names.loc[unique_column_names <= MAX_UNIQUE_COLUMN_VALUES].index.tolist()

    # initialize an empty dict that will store results in the loop
    interesting_columns_with_few_unique = {}

    # loop through the column names that have the fewest unique values
    for colname in column_names_with_few_unique_values:
        # create/assign a value
        xxx = df_opps[colname].unique().tolist()
        yyy = {}
        #yyy['_Non-String_'] = 0
        for x in xxx:
            if isinstance(x, str):
                yyy[x] = len(df_opps[df_opps[colname] == x].index)

            #elif isinstance(x,pd._libs.missing.NAType):
            # elif isinstance(x,pd.NA):
            #     yyy['pd._libs.missing.NAType'] = -1

            #elif x is None or isinstance(x,pd.NA):
                #yyy['NoneType_aka_None'] += df_opps[colname].isnull().sum()


            #else:
            #    print(f'Failed type: {x} type: {type(x)}')
            #    yyy['_Non-String_'] += 1
                #yyy[x] = -3 # df_opps[colname].isna()
                # yyy[x] = df_opps[colname].isnull().sum()



        yyy['_None_'] = df_opps[colname].isnull().sum()
        # yyy['_NaT_']
        interesting_columns_with_few_unique[colname] = yyy

    # display the result
    return interesting_columns_with_few_unique

# def get_stratified_sample(df_opps: pd.DataFrame, percentage) -> pd.DataFrame:
#     """Docstring TBD"""
#     # n_samples = int(df_opps.shape[0] * percentage)
#     res = df_opps.groupby('Won', group_keys=False).apply(lambda x: x.sample(frac=percentage))
#     return res


# def get_stratified_sample2(df_opps: pd.DataFrame, percentage) -> pd.DataFrame:
#     """Docstring TBD"""
#     stratify_list = []
#     #df_train, df_test = train_test_split(df_opps, test_size=percentage, stratify=df_opps[["Segment", "Insert"]])
#     df_train, df_test = train_test_split(df_opps, test_size=percentage, stratify=)
#     return df_train, df_test
