import pandas as pd
import logging
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from conf.config import TARGET_COL, TEST_SIZE, RANDOM_STATE

# Custom Transformer
class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        base_col_name = X_transformed.columns[0]
        year_col_name = base_col_name + '_year'
        month_col_name = base_col_name + '_month'
        day_col_name = base_col_name + '_day'
        # Assuming X is a DataFrame and the column to be split is 'date'
        X_transformed[year_col_name] = pd.DatetimeIndex(X[base_col_name]).year
        X_transformed[month_col_name] = pd.DatetimeIndex(X[base_col_name]).month
        X_transformed[day_col_name] = pd.DatetimeIndex(X[base_col_name]).day
        X_transformed.drop(base_col_name, axis=1, inplace=True)

        #store feature names
        self.feature_names = X_transformed.columns

        return X_transformed

    def set_output(self, *, transform=None):
        super().set_output(transform=transform)
        return self

    def get_feature_names_out(self, input_features=None):
        return self.feature_names


def create_feature_engineering_pipeline():
    """
    Creates a feature engineering pipeline using ColumnTransformer
    with pipelines for different types of feature transformations
    like OneHotEncoder, SimpleImputer etc.

    Parameters:
    df (pd.DataFrame): The training data as a pandas DataFrame.

    Returns:
    ColumnTransformer: A configured ColumnTransformer object.
    """

    # OHE to encode categorical columns
    enc_ohe = OneHotEncoder(handle_unknown='ignore')
    # custom transformer to split one date time column into its constituting elements (data, month and year)
    trans_date = DateTransformer()
    # imputer to replace nulls with the most frequent value
    enc_imputer_freq = SimpleImputer(strategy='most_frequent')

    # preprocessing pipeline to perform OHE
    col_trans_cat_01 = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
                        'affiliate_provider', 'signup_app', 'first_device_type', 'first_browser']
    pipe_trans_cat_01 = Pipeline(
        steps=[
            ('encoding', enc_ohe)
        ]
    )

    # preprocessing pipeline to perform frequent value imputation
    col_age = ['age']
    pipe_trans_age = Pipeline(
        steps=[
            ('imputing', enc_imputer_freq)
        ]
    )

    # preprocessing pipeline to perform imputation first and then OHE
    col_fat = ['first_affiliate_tracked']
    pipe_trans_fat = Pipeline(
        steps=[
            ('imputing', enc_imputer_freq),
            ('encoding', enc_ohe)
        ]
    )

    # preprocessing pipeline to perform custom processing (column split)
    col_trans_date_01 = ['date_account_created']
    pipe_trans_date_01 = Pipeline(
        steps=[
            ('datesplit', trans_date)
        ]
    )

    # stitching the full pipeline together as a ColumnTransformer
    ctrans_preproc_01 = ColumnTransformer(
        transformers=[
            ('tcat_01', pipe_trans_cat_01, col_trans_cat_01),
            ('tage', pipe_trans_age, col_age),
            ('tfat', pipe_trans_fat, col_fat),
            ('tdate_01', pipe_trans_date_01, col_trans_date_01)
        ],
        remainder='passthrough'
    )

    return ctrans_preproc_01

def do_data_prep(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # features to include all but last column, which usually is the target class
    cols_raw_features = df_train.columns[:-1]
    cols_insignificant = ['id', 'date_first_booking']
    X_full = pd.concat((df_train[cols_raw_features], df_test), axis=0, ignore_index=True).copy()
    # Updating age outliers to -1
    v_age = X_full.age.values
    X_full['age'] = np.where(np.logical_or(v_age < 14, v_age > 100), -1, v_age)
    # exclude test data set
    X = X_full[:df_train.shape[0]][cols_raw_features]
    y = df_train[TARGET_COL]
    X_test = X_full[df_train.shape[0]:][cols_raw_features]

    # label encode the target variable
    enc_le = LabelEncoder()
    tmp_y = y.values
    y = enc_le.fit_transform(tmp_y.ravel())

    # This code splits the data into training and validation sets, then drops the insignificant columns from the full dataset, training set, validation set, and test set.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # X_full = X_full.drop(cols_insignificant, axis=1)
    X_train = X_train.drop(cols_insignificant, axis=1)
    X_val = X_val.drop(cols_insignificant, axis=1)
    X_test = X_test.drop(cols_insignificant, axis=1)

    return X_train, X_val, X_test, y_train, y_val