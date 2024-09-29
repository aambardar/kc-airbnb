# main.py
import traceback
import logger_setup
from data_loader import DataLoader
import feature_engg, model
from src.feature_engg import create_feature_engineering_pipeline
from conf.config import PATH_DATA

def main():
    try:
        logger_setup.logger.info("START ...")
        # Load data
        data_loader = DataLoader(PATH_DATA)
        df_train_users, df_train_sessions, df_test_users = data_loader.load_data()

        # Engineer features
        X_train, X_val, X_test, y_train, y_val = feature_engg.do_data_prep(df_train_users, df_test_users)
        preproc = create_feature_engineering_pipeline()

        # Tune hyperparameters
        study= model.run_hyperparam_tuning(X_train, y_train, preproc)
        best_performing_trial = study.best_trial

        logger_setup.logger.info("... FINISH")
    except Exception as e:
        logger_setup.logger.error("An error occurred during the execution")
        logger_setup.logger.error(str(e))
        logger_setup.logger.error(traceback.format_exc())
    finally:
        logger_setup.logger.info('-' * 80)  # Add a horizontal line after successful execution


if __name__ == "__main__":
    logger_setup.setup_logging()
    main()