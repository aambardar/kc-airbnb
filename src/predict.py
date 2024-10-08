import numpy as np
import pandas as pd
from conf.config import MODEL_VERSION, PATH_OUT_PREDICTIONS
import logger_setup

def predict(best_models_dict, test_data):
    logger_setup.logger.debug("START ...")
    predictions_dict = {}

    for key, value in best_models_dict.items():
        logger_setup.logger.info(f'Predicting using {key} model.')
        y_pred = value.predict_proba(test_data)
        predictions_dict[key] = y_pred
    logger_setup.logger.debug("... FINISH")
    return predictions_dict

def submit_predictions(predictions_dict, test_data_ids, label_encoder, version_to_use):
    logger_setup.logger.debug("START ...")
    for key, value in predictions_dict.items():
        logger_setup.logger.info(f'Predicting using {key} model.')
        ids = []  # list of ids
        cts = []  # list of countries
        for i in range(len(test_data_ids)):
            idx = test_data_ids[i]
            ids += [idx] * 5
            cts += label_encoder.inverse_transform(np.argsort(value[i])[::-1])[:5].tolist()
        submission = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
        logger_setup.logger.info(f'Writing submission file: {PATH_OUT_PREDICTIONS}sub_{key}_{version_to_use}.csv')
        submission.to_csv(f'{PATH_OUT_PREDICTIONS}sub_{key}_{version_to_use}.csv', index=False)
    logger_setup.logger.debug("... FINISH")