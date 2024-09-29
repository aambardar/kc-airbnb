# main.py
import traceback
import logger_setup
from data_loader import DataLoader
import feature_engg, model
from src.feature_engg import create_feature_engineering_pipeline
from conf.config import PATH_DATA

def main():
    try:
        logger_setup.logger.debug("START ...")
        # Load data
        data_loader = DataLoader(PATH_DATA)
        df_train_users, df_train_sessions, df_test_users = data_loader.load_data()

        # Engineer features
        X_train, X_val, X_test, y_train, y_val = feature_engg.do_data_prep(df_train_users, df_test_users)
        preproc = create_feature_engineering_pipeline()

        # Tune hyperparameters
        study, best_models_dict = model.run_hyperparam_tuning(X_train, y_train, preproc)
        model.analyse_optuna_study(study)
        model.save_artefacts(study, best_models_dict)
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