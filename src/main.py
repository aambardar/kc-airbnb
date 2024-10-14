# main.py
import traceback
import logger_setup
import re
import time
import joblib
import pandas as pd
from data_loader import DataLoader
import feature_engg, model, predict, utility
from src.feature_engg import create_feature_engineering_pipeline
from conf.config import PATH_DATA, BYPASS_TRAINING, PATH_OUT_MODELS, BYPASS_TRAINING_VERSION, MODEL_VERSION, \
    IS_SPLIT_NEEDED, PATH_OUT_ARTEFACTS, INTERIM_DATA_SESSIONS, INTERIM_DATA_TRAIN, INTERIM_DATA_TEST, INTERIM_DATA_LABELS, TARGET_COL


def load_raw_data():
    logger_setup.logger.debug("START ...")
    data_loader = DataLoader(PATH_DATA)
    df_train_users, df_train_sessions, df_test_users = data_loader.load_data()
    logger_setup.logger.debug("... FINISH")
    return df_train_users, df_train_sessions, df_test_users

def prepare_sessions_data(sessions_raw_data):
    logger_setup.logger.debug("START ...")
    # Categorical values that appear less than the threshold will be removed.
    VALUE_THRESHOLD = 0.005
    CATEGORICAL_FEATURES = ['action', 'action_type', 'action_detail', 'device_type']
    SECS_ELAPSED_NUMERICAL = 'secs_elapsed'
    INDEX_COLUMN = 'user_id'

    logger_setup.logger.info(f'Shape of input data frame is:{sessions_raw_data.shape}')
    sessions_raw_data.set_index(INDEX_COLUMN, inplace=True)
    sessions_raw_data.fillna(-1, inplace=True)
    # Extract features from sessions.
    utility.remove_rare_values_inplace(sessions_raw_data, CATEGORICAL_FEATURES, VALUE_THRESHOLD)
    frequency_counts = utility.extract_frequency_counts(sessions_raw_data, CATEGORICAL_FEATURES)
    simple_stats = utility.extract_distribution_stats(sessions_raw_data, SECS_ELAPSED_NUMERICAL)
    # Save new data.
    session_data = pd.concat((frequency_counts, simple_stats), axis=1)
    session_data.fillna(-1, inplace=True)
    logger_setup.logger.info(f'Shape of output data frame before saving is:{session_data.shape}')
    session_data.to_csv(f'{PATH_OUT_ARTEFACTS}{INTERIM_DATA_SESSIONS}')
    logger_setup.logger.debug("... FINISH")

def prepare_user_data(training_data, testing_data):
    logger_setup.logger.debug("START ...")
    INDEX_COLUMN = 'id'
    CATEGORICAL_FEATURES = ['affiliate_channel', 'affiliate_provider',
                            'first_affiliate_tracked', 'first_browser',
                            'first_device_type', 'gender', 'language', 'signup_app',
                            'signup_method', 'signup_flow']
    ACCOUNT_DATE = 'date_account_created'
    UNUSED_DATE_COLUMNS = ['timestamp_first_active', 'date_first_booking']
    # A parameter to speed-up computation. Categorical values that appear
    # less than the threshold will be removed.
    VALUE_THRESHOLD = 0.001
    DATE_FORMAT = '%Y-%m-%d'  # Expected format for date.

    training_data.set_index(INDEX_COLUMN, inplace=True)
    testing_data.set_index(INDEX_COLUMN, inplace=True)

    labels = training_data[TARGET_COL].copy()
    training_data.drop(TARGET_COL, inplace=True, axis=1)
    features = pd.concat((training_data, testing_data), axis=0)
    features.fillna(-1, inplace=True)

    # Process all features by removing rare values, appling one-hot-encoding to
    # those that are categorical and extracting numericals from ACCOUNT_DATE.

    utility.remove_rare_values_inplace(features, CATEGORICAL_FEATURES, VALUE_THRESHOLD)
    features = utility.apply_one_hot_encoding(features, CATEGORICAL_FEATURES)
    utility.extract_dates_inplace(features, ACCOUNT_DATE, DATE_FORMAT)
    features.drop(UNUSED_DATE_COLUMNS, inplace=True, axis=1)
    logger_setup.logger.debug("... FINISH")
    return features, labels, training_data.index, testing_data.index

def prepare_full_data(df_train_users, df_train_sessions, df_test_users):
    logger_setup.logger.debug("START ...")
    # if is_split_needed is False X_train and X_val are the same (likewise, y_train and y_val will be the same).
    X_train, X_val, X_test, y_train, y_val, enc_label = feature_engg.do_data_prep(df_train_users, df_test_users,
                                                                                  is_split_needed=IS_SPLIT_NEEDED)
    logger_setup.logger.debug("... FINISH")
    return X_train, X_val, X_test, y_train, y_val, enc_label

def run_hyperparam_tuning(X_train, y_train):
    logger_setup.logger.debug("START ...")
    if BYPASS_TRAINING:
        logger_setup.logger.info('<<< TRAINING BYPASSED >>>')
        best_models_pipe_dict = {}
        file_pattern = r'^best_model_pipe_.*' + re.escape(BYPASS_TRAINING_VERSION) + r'\.pkl$'
        best_model_files = utility.pick_files_by_pattern(PATH_OUT_MODELS, file_pattern)
        for index, value in enumerate(best_model_files):
            logger_setup.logger.info(f'Loading file: {value} from location: {PATH_OUT_MODELS}')
            model_pipe = joblib.load(f'{PATH_OUT_MODELS}{value}')
            best_models_pipe_dict[value] = model_pipe
    else:
        logger_setup.logger.info('<<< TRAINING REQUESTED >>>')
        preproc = create_feature_engineering_pipeline()
        # Tune hyperparameters
        study, best_models_pipe_dict = model.run_hyperparam_tuning(X_train, y_train, preproc)
        model.analyse_optuna_study(study)
        # Saving tuned model
        model.save_artefacts(study, best_models_pipe_dict)
    logger_setup.logger.debug("... FINISH")

def run_inference(best_models_pipe_dict, train, test, test_ids, label_encoder):
    # Retrieving feature names
    features = model.get_feature_names(best_models_pipe_dict, train.columns)
    predictions_dict = predict.predict(best_models_pipe_dict, test)
    predict.submit_predictions(predictions_dict, test_ids, label_encoder,
                               BYPASS_TRAINING_VERSION if BYPASS_TRAINING else MODEL_VERSION)
    logger_setup.logger.debug("... FINISH")

def main():
    start_time = time.time()
    try:
        logger_setup.logger.debug("START ...")
        # Load raw data
        df_raw_users, df_raw_sessions, df_raw_test_users = load_raw_data()
        logger_setup.logger.info(f'>>> Loading of raw data completed.')
        # Prepare sessions data
        prepare_sessions_data(df_raw_sessions)
        logger_setup.logger.info(f'>>> SESSIONS data processing completed.')
        features, labels, training_ids, testing_ids = prepare_user_data(df_raw_users, df_raw_test_users)
        logger_setup.logger.info(f'>>> USER data processing completed.')

        sessions_features = pd.read_csv(f'{PATH_OUT_ARTEFACTS}{INTERIM_DATA_SESSIONS}', index_col=0)
        features = pd.concat((features, sessions_features), axis=1)
        features.fillna(-1, inplace=True)
        # Save data training and testing data.
        training = features.ix[training_ids]
        testing = features.ix[testing_ids]

        # Warning: When saving the data, it's important that the header is True,
        # because labels is of type pandas.core.series.Series, while training is of
        # type pandas.core.frame.DataFrame, and they have different default values
        # for the header argument.

        assert set(training.index) == set(labels.index)
        training.to_csv(f'{PATH_OUT_ARTEFACTS}{INTERIM_DATA_TRAIN}', header=True)
        testing.to_csv(f'{PATH_OUT_ARTEFACTS}{INTERIM_DATA_TEST}', header=True)
        labels.to_csv(f'{PATH_OUT_ARTEFACTS}{INTERIM_DATA_LABELS}', header=True)
        logger_setup.logger.info(f'>>> Data processing completed.')

        """ Perform prediction. """
        train_df = pd.read_csv(f'{PATH_OUT_ARTEFACTS}{INTERIM_DATA_TRAIN}', index_col=0)
        labels_df = pd.read_csv(f'{PATH_OUT_ARTEFACTS}{INTERIM_DATA_LABELS}', index_col=0)
        test_df = pd.read_csv(f'{PATH_OUT_ARTEFACTS}{INTERIM_DATA_TEST}', index_col=0)
        assert set(train_df.index) == set(labels_df.index)

        model.prepare_for_prediction(train_df, test_df, labels_df, TARGET_COL)
        logger_setup.logger.info(f'>>> Prediction submission completed.')
        logger_setup.logger.debug("... FINISH")
    except Exception as e:
        logger_setup.logger.error("An error occurred during the execution")
        logger_setup.logger.error(str(e))
        logger_setup.logger.error(traceback.format_exc())
    finally:
        end_time = time.time()
        logger_setup.logger.debug('-' * 80)  # Add a horizontal line after every execution
        logger_setup.logger.debug(f'Total Execution Time:{(end_time - start_time) / 60:.2f} mins.')
        logger_setup.logger.debug('-' * 80)  # Add a horizontal line after every execution

def main_old():
    start_time = time.time()
    try:
        logger_setup.logger.debug("START ...")
        # Load data
        data_loader = DataLoader(PATH_DATA)
        df_train_users, df_train_sessions, df_test_users = data_loader.load_data()

        # Engineer features
        # if is_split_needed is False X_train and X_val are the same (likewise, y_train and y_val will be the same).
        X_train, X_val, X_test, y_train, y_val, enc_label = feature_engg.do_data_prep(df_train_users, df_test_users, is_split_needed=IS_SPLIT_NEEDED)

        if BYPASS_TRAINING:
            logger_setup.logger.info('<<< TRAINING BYPASSED >>>')
            best_models_pipe_dict = {}
            file_pattern = r'^best_model_pipe_.*' + re.escape(BYPASS_TRAINING_VERSION) + r'\.pkl$'
            best_model_files = utility.pick_files_by_pattern(PATH_OUT_MODELS, file_pattern)
            for index, value in enumerate(best_model_files):
                logger_setup.logger.info(f'Loading file: {value} from location: {PATH_OUT_MODELS}')
                model_pipe = joblib.load(f'{PATH_OUT_MODELS}{value}')
                best_models_pipe_dict[value] = model_pipe
        else:
            logger_setup.logger.info('<<< TRAINING REQUESTED >>>')
            preproc = create_feature_engineering_pipeline()
            # Tune hyperparameters
            study, best_models_pipe_dict = model.run_hyperparam_tuning(X_train, y_train, preproc)
            model.analyse_optuna_study(study)
            # Saving tuned model
            model.save_artefacts(study, best_models_pipe_dict)

        # Retrieving feature names
        features = model.get_feature_names(best_models_pipe_dict, X_train.columns)
        predictions_dict = predict.predict(best_models_pipe_dict, X_test)
        predict.submit_predictions(predictions_dict, df_test_users.id, enc_label, BYPASS_TRAINING_VERSION if BYPASS_TRAINING else MODEL_VERSION)
        logger_setup.logger.debug("... FINISH")
    except Exception as e:
        logger_setup.logger.error("An error occurred during the execution")
        logger_setup.logger.error(str(e))
        logger_setup.logger.error(traceback.format_exc())
    finally:
        end_time = time.time()
        logger_setup.logger.debug('-' * 80)  # Add a horizontal line after every execution
        logger_setup.logger.debug(f'Total Execution Time:{(end_time - start_time) / 60:.2f} mins.')
        logger_setup.logger.debug('-' * 80)  # Add a horizontal line after every execution


if __name__ == "__main__":
    logger_setup.setup_logging()
    main()