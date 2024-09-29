# data_loader.py
import pandas as pd
import logger_setup

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self):
        try:
            logger_setup.logger.info("START ...")
            logger_setup.logger.info(f'Loading data from path: {self.data_path}')
            df_train_users = pd.read_csv(f'{self.data_path}abnb_train_users.csv')
            df_train_sessions = pd.read_csv(f'{self.data_path}abnb_sessions.csv')
            df_test_users = pd.read_csv(f'{self.data_path}abnb_test_users.csv')
            logger_setup.logger.info(f'Loaded abnb_train_users with shape: {df_train_users.shape}')
            logger_setup.logger.info(f'Loaded abnb_sessions with shape: {df_train_sessions.shape}')
            logger_setup.logger.info(f'Loaded abnb_test_users with shape: {df_test_users.shape}')
            logger_setup.logger.info(f'Successfully loaded data from path: {self.data_path}')
            logger_setup.logger.info("... FINISH")
            return df_train_users, df_train_sessions, df_test_users
        except Exception as e:
            logger_setup.logger.error(f'Failed to load data from {self.data_path}: {str(e)}')
            raise