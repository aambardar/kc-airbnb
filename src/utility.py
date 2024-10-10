import logger_setup
import pandas as pd
import re, os

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
    logger_setup.logger.info('START ...')
    new_columns = [col if col.lower().startswith(prefix) else prefix + col.lower() for col in df.columns]
    df.columns = new_columns
    logger_setup.logger.info('... FINISH')
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
    # Ensure input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Ensure columns_dict is a dictionary
    if not isinstance(columns_dict, dict):
        raise ValueError("columns_dict should be a dictionary")

    # Rename columns
    new_df = df.rename(columns=columns_dict)

    return new_df