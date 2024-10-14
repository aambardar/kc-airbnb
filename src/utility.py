import logger_setup
import pandas as pd
import numpy as np
import collections
import re, os, datetime as dt
import scipy.stats as stats

def pick_files_by_pattern(directory, pattern):
    logger_setup.logger.debug("START ...")
    # Compile the regex pattern
    regex = re.compile(pattern)

    # List all files that match the regex pattern
    matched_files = [filename for filename in os.listdir(directory) if regex.match(filename)]
    logger_setup.logger.info(f'Files matching with the pattern are:\n{matched_files}')
    logger_setup.logger.debug("... FINISH")
    return matched_files

def add_prefix_to_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    logger_setup.logger.debug('START ...')
    new_columns = [col if col.lower().startswith(prefix) else prefix + col.lower() for col in df.columns]
    df.columns = new_columns
    logger_setup.logger.debug('... FINISH')
    return df


def rename_columns(df, columns_dict):
    """
    Renames columns of the DataFrame based on the provided dictionary.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    columns_dict (dict): A dictionary where keys are current column names and values are new column names.

    Returns:
    pd.DataFrame: A new DataFrame with the renamed columns.
    """
    logger_setup.logger.debug('START ...')
    # Ensure input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Ensure columns_dict is a dictionary
    if not isinstance(columns_dict, dict):
        raise ValueError("columns_dict should be a dictionary")

    # Rename columns
    new_df = df.rename(columns=columns_dict)
    logger_setup.logger.debug('... FINISH')
    return new_df

def remove_rare_values_inplace(df_frame, column_list, threshold):
    """ Remove rare values to speed up computation.

    Args:
        df_frame -- A pandas data frame.
        column_list -- A list of columns.
        threshold -- The threshold, below which a value is removed.
    """
    logger_setup.logger.debug('START ...')
    insignificant_population = int(np.floor(threshold * len(df_frame)))
    for cat in column_list:
        freqs = collections.Counter(df_frame[cat])
        other = [i for i in freqs if freqs[i] < insignificant_population]
        for i in other:
            df_frame[cat].replace(i, 'other', inplace=True)
    logger_setup.logger.debug('... FINISH')

def apply_one_hot_encoding(pd_frame, column_list):
    """ Apply One-Hot-Encoding to pd_frame's categorical columns.

    Args:
        df_frame -- A pandas data frame.
        column_list -- A list of categorical columns, in df_frame.

    Returns:
        A pandas dataframe where the colums in column_list have been replaced
            by one-hot-encoded-columns.
    """
    logger_setup.logger.debug('START ...')
    new_column_list = []
    for col in column_list:
        tmp = pd.get_dummies(pd_frame[col], prefix=col)
        new_column_list.append(tmp)
    new_pd_frame = pd.concat(new_column_list+[pd_frame], axis=1)
    new_pd_frame.drop(column_list, inplace=True, axis=1)
    logger_setup.logger.debug('... FINISH')
    return new_pd_frame

def _parse_date(date_str, format_str):
    """ Extract features from the data_account_creted column.

    Warning: There is strong dependency between this method and the method
    replace_dates_inplace.

    Args:
        date_str -- A string containing a date value.
        str_format -- The format of the string date.

    Returns:
        A list of 4 values containing the extracted [year, month, day, weekend].
    """
    time_dt = dt.datetime.strptime(date_str, format_str)
    return [time_dt.year, time_dt.month, time_dt.day, time_dt.weekday()]

def extract_dates_inplace(features, date_column, date_format):
    """ Extract from the date-columns, year, month, and other numericals.

    Warning: There is strong dependency between this method and _parse_date.
    """
    extracted_vals = np.vstack(features[date_column].apply(
        (lambda x: _parse_date(x, date_format))))
    for i, period in enumerate(['year', 'month', 'day', 'weekday']):
        features['%s_%s' % (date_column, period)] = extracted_vals[:, i]
    features.drop(date_column, inplace=True, axis=1)

def extract_frequency_counts(pd_frame, column_list):
    """ Extract frequency counts from pd_frame.

    For each index (that correspond to a user) this method will count the
    number of times that C == Ci, where C is a column in column_list, and Ci
    is a unique value of that column. The arg column_list is assumed
    to contain categorical columns.

    Args:
        df_frame -- A pandas data frame.
        column_list -- A list of columns.

    Returns:
        A pandas DataFrame, containing frequency counts.
    """
    logger_setup.logger.debug('START ...')
    df_extracted_sessions = []
    for col in column_list:
        for val in set(pd_frame[col]):
            logger_setup.logger.info(f'Extracting frequency counts for ({col} == {val})')
            tmp_df = pd_frame.groupby(pd_frame.index).apply(
                lambda group, x=col, y=val: np.sum(group[x] == y))
            tmp_df.name = '%s=%s' % (col, val)
            df_extracted_sessions.append(tmp_df)
    frequency_counts = pd.concat(df_extracted_sessions, axis=1)
    logger_setup.logger.debug('... FINISH')
    return frequency_counts

def extract_distribution_stats(pd_frame, numerical_col):
    """ Extract simple distribution statistics from a numerical column.

    Args:
        df_frame -- A pandas data frame.
        numerical_col -- A column in pd_frame that contains numerical values.

    Returns:
        A pandas DataFrame, containing simple satistics for col_name.
    """
    logger_setup.logger.debug('START ...')
    tmp_df = pd_frame[numerical_col].groupby(pd_frame.index).aggregate(
        [np.mean, np.std, np.median, stats.skew])
    tmp_df.columns = ['%s_%s'% (numerical_col, i) for i in tmp_df.columns]
    logger_setup.logger.debug('... FINISH')
    return tmp_df