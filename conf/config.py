import os

# OS path configuration
PATH_DATA = '../data/'
PATH_CONTENT = '../content/'
PATH_SRC = '../src/'
PATH_OUTPUT = '../out/'
PATH_OUT_LOGS = '../out/logs/'
PATH_OUT_MODELS = '../out/models/'
PATH_OUT_PREDICTIONS = '../out/predictions/'
PATH_OUT_VISUALS = '../out/visualisations/'

# Logging configurations
LOG_FILE = os.path.join(PATH_OUT_LOGS, 'application.log')
LOG_ROOT_LEVEL = 'DEBUG'
LOG_FILE_LEVEL = 'DEBUG'
LOG_CONSOLE_LEVEL = 'ERROR'

# Stylesheet configurations
MPL_STYLE_FILE = os.path.join(PATH_CONTENT, 'custom_mpl_stylesheet.mplstyle')

# Feature Engineering configuration
NUMERICAL_IMPUTATION_STRATEGY = 'mean'  # Options: 'mean', 'median', 'most_frequent'
CATEGORICAL_IMPUTATION_STRATEGY = 'most_frequent'  # Options: 'most_frequent', 'constant'

# Path for saving models
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILENAME = 'trained_model.pkl'
BEST_MODEL_PATH = os.path.join(MODEL_PATH, MODEL_FILENAME)

# Other configurations
BYPASS_TRAINING = False
RANDOM_STATE = 43
TEST_SIZE = 0.2  # For train-test split
OPTUNA_TRIAL_COUNT = 10
MODEL_VERSION = '013'
BYPASS_TRAINING_VERSION = '009'
TARGET_COL = 'country_destination'