# main.py
import traceback
import logger_setup
import re
import joblib
from data_loader import DataLoader
import feature_engg, model, predict, utility
from src.feature_engg import create_feature_engineering_pipeline
from conf.config import PATH_DATA, BYPASS_TRAINING, PATH_OUT_MODELS, BYPASS_TRAINING_VERSION, MODEL_VERSION


def main():
    try:
        logger_setup.logger.debug("START ...")
        # Load data
        data_loader = DataLoader(PATH_DATA)
        df_train_users, df_train_sessions, df_test_users = data_loader.load_data()

        # Engineer features
        # if is_split_needed is False X_train and X_val are the same (likewise, y_train and y_val will be the same).
        X_train, X_val, X_test, y_train, y_val, enc_label = feature_engg.do_data_prep(df_train_users, df_test_users, is_split_needed=True)

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
        logger_setup.logger.debug('-' * 80)  # Add a horizontal line after successful execution


if __name__ == "__main__":
    logger_setup.setup_logging()
    main()