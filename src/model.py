import numpy as np
import optuna
import time
import pandas as pd
from conf.config import OPTUNA_TRIAL_COUNT, RANDOM_STATE, PATH_OUT_VISUALS, MODEL_VERSION, PATH_OUT_MODELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import logger_setup
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import utility
import joblib
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_param_importances,
    plot_edf
)

def ndcg_at_5(y_true, y_pred_proba, k=5):
    logger_setup.logger.debug("START ...")
    # Convert y_true to a binary relevance array
    y_true_binary = np.zeros_like(y_pred_proba)
    y_true_binary[np.arange(len(y_true)), y_true] = 1

    # Sort by predicted probabilities
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]

    # Compute DCG
    gains = y_true_binary[np.arange(len(y_true))[:, None], top_k_indices]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts, axis=1)

    # Compute IDCG
    ideal_gains = np.sort(y_true_binary, axis=1)[:, -k:][:, ::-1]
    idcg = np.sum(ideal_gains / discounts, axis=1)

    # Compute NDCG
    ndcg = np.mean(dcg / (idcg + 1e-10))
    logger_setup.logger.debug("... FINISH")
    return ndcg

def run_hyperparam_tuning(X_train, y_train, col_trans):
    logger_setup.logger.debug("START ...")
    models = {}
    def optuna_objective(trial):
        logger_setup.logger.debug("START ...")

        classifier_models = trial.suggest_categorical('model', ['xgb', 'rfc'])
        logger_setup.logger.info(f'Trial: {trial}')
        logger_setup.logger.info(f'Trial number: {trial.number}')

        if classifier_models == 'rfc':
            # n_estimators: Specifies the number of decision trees in the RandomForestClassifier. A larger number of trees can improve model performance by reducing variance, but also increases computation time/complexity.
            rfc_n_estimators = trial.suggest_int('rfc_n_estimators', 50, 300)
            # max_depth: Sets the maximum depth for each decision tree in the RandomForestClassifier. Limiting depth helps prevent overfitting by controlling tree size/complexity, but shallow trees may lead to underfitting.
            rfc_max_depth = trial.suggest_int('rfc_max_depth', 1, 10)
            # criterion: Determines the function to measure the quality of a split in the RandomForestClassifier. Common options are "gini" for the Gini impurity and "entropy" for the information gain.
            rfc_criterion = trial.suggest_categorical('rfc_criterion', ['gini', 'entropy'])
            # min_samples_split: Minimum samples needed to split an internal node in the RandomForestClassifier. Higher values reduce overfitting but may cause underfitting by limiting splits.
            rfc_min_samples_split = trial.suggest_float('rfc_min_samples_split', 0.01, 1.0)

            model_rfc = RandomForestClassifier(n_estimators=rfc_n_estimators, max_depth=rfc_max_depth,
                                               criterion=rfc_criterion, min_samples_split=rfc_min_samples_split)
            pipe_model = Pipeline(
                steps=[
                    ('preprocessing', c_trans),
                    ('modelling', model_rfc)
                ]
            )
        elif classifier_models == 'knn':
            # n_neighbors: Number of neighbors to consider in KNN. Affects the bias-variance trade-off, influencing model complexity and accuracy.
            knn_n_neighbors = trial.suggest_int('knn_n_neighbors', 1, 5)
            # weights: Determines neighbor influence in KNN. 'Uniform' weights all points equally; 'distance' assigns more weight to closer neighbors. Affects decision boundaries and performance.
            knn_weights = trial.suggest_categorical('knn_weights', ['uniform', 'distance'])
            # p: Determines the power parameter for the Minkowski distance metric in KNN. p=1 uses Manhattan distance, while p=2 uses Euclidean distance.
            knn_p = trial.suggest_int('knn_p', 1, 2)

            model_knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors, weights=knn_weights, p=knn_p)
            pipe_model = Pipeline(
                steps=[
                    ('preprocessing', c_trans),
                    ('modelling', model_knn)
                ]
            )
        elif classifier_models == 'xgb':
            # n_estimators: Number of boosting rounds in XGBClassifier. More estimators can improve model performance but may increase training time and risk overfitting.
            xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 10, 100)
            # learning_rate: Shrinks the contribution of each boosting round in XGBClassifier. Lower values improve model robustness but require more estimators.
            xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 1e-5, 10)
            # max_depth: Maximum depth of each tree in XGBClassifier. Controls model complexity; deeper trees can capture more patterns but may lead to overfitting.
            xgb_max_depth = trial.suggest_int('xgb_max_depth', 2, 10)
            # subsample: Proportion of training data sampled for each boosting round in XGBClassifier. Prevents overfitting; lower values increase model robustness but may reduce performance.
            xgb_subsample = trial.suggest_float('xgb_subsample', 0.5, 0.9)
            # colsample_bytree: Fraction of features sampled for each tree in XGBClassifier. Controls overfitting; lower values increase model robustness by introducing randomness in feature selection.
            xgb_colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.3, 0.9)
            # objective: Defines the learning task and the corresponding loss function in XGBClassifier. Common values include 'binary:logistic' for binary classification and 'multi:softprob' for multi-class classification.
            xgb_objective = trial.suggest_categorical('xgb_objective', ['multi:softprob'])

            model_xgb = xgb.XGBClassifier(n_estimators=xgb_n_estimators, learning_rate=xgb_learning_rate,
                                          max_depth=xgb_max_depth, subsample=xgb_subsample,
                                          colsample_bytree=xgb_colsample_bytree, objective=xgb_objective, device='cuda')
            pipe_model = Pipeline(
                steps=[
                    ('preprocessing', c_trans),
                    ('modelling', model_xgb)
                ]
            )
        else:
            # C: Inverse of regularization strength in LogisticRegression. Smaller values specify stronger regularization, controlling model complexity to prevent overfitting.
            lr_C = trial.suggest_float('C', 1e-5, 1000)
            # max_iter: Maximum number of iterations for the solver in LogisticRegression. Ensures convergence and model accuracy during training by setting an upper limit on iterations.
            lr_max_iter = trial.suggest_int('max_iter', 10, 1000)
            # fit_intercept: Determines whether to include an intercept term in the LogisticRegression model. True adds an intercept; False assumes data is already centered.
            lr_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])

            model_lr = LogisticRegression(C=lr_C, max_iter=lr_max_iter, fit_intercept=lr_fit_intercept)
            pipe_model = Pipeline(
                steps=[
                    ('preprocessing', c_trans),
                    ('modelling', model_lr)
                ]
            )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

        # Perform cross-validation
        ndcg_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]
            pipe_model.fit(X_tr, y_tr)
            y_vl_pred_proba = pipe_model.predict_proba(X_vl)
            ndcg_scores.append(ndcg_at_5(y_vl, y_vl_pred_proba, k=5))

        scores = ndcg_scores
        score = np.mean(scores)
        # Print the results
        logger_setup.logger.info(f'NDCG@5 scores: {scores}')
        logger_setup.logger.info(f'Mean score: {score}')

        models[trial.number] = pipe_model
        logger_setup.logger.debug("... FINISH")
        return score

    # creation of Optuna study
    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE))
    X_train = X_train
    y_train = y_train
    c_trans = col_trans
    # optimise the study
    study.optimize(optuna_objective, n_trials=OPTUNA_TRIAL_COUNT)
    end_time = time.time()
    elapsed_time = end_time - start_time
    best_performing_trial = study.best_trial
    study_full_metrics = study.trials_dataframe()
    # grouping the metrics by model type (params_model), and then use the idxmax method to find the index of the row with the best model performance (value) for each group
    study_best_models = study_full_metrics.loc[study_full_metrics.groupby('params_model')['value'].idxmax()]
    # retrieve trial number of best model for each model type - the Optuna metrics dataframe index and trial number are the same.
    best_rfc_trial = study_best_models[study_best_models['params_model'] == 'rfc']['number'].values[0]
    best_xgb_trial = study_best_models[study_best_models['params_model'] == 'xgb']['number'].values[0]
    model_pipe_best = models[best_performing_trial.number]
    model_pipe_best_rfc = models[best_rfc_trial]
    model_pipe_best_xgb = models[best_xgb_trial]
    best_models_dict = {
        'overall': model_pipe_best,
        'rfc': model_pipe_best_rfc,
        'xgb': model_pipe_best_xgb
    }
    logger_setup.logger.info(f'Best trial was at number {best_performing_trial.number} with params as:\n {best_performing_trial.params}')
    logger_setup.logger.info(f'Best score value is: {best_performing_trial.value}')

    # fetch number of trial runs per model type
    num_rfc_trials = study_full_metrics[study_full_metrics['params_model'] == 'rfc'].shape[0]
    num_xgb_trials = study_full_metrics[study_full_metrics['params_model'] == 'xgb'].shape[0]
    logger_setup.logger.info(f'\nTotal trials = {num_rfc_trials + num_xgb_trials}\n-- RFC trials = {num_rfc_trials}\n-- XGB trials = {num_xgb_trials} \nTotal time elapsed during the Optuna run = {elapsed_time/60:.2f} minutes')

    logger_setup.logger.debug("... FINISH")
    return study, best_models_dict

# Define a function to save plots
def save_plot(plot_func, study, filename):
    logger_setup.logger.debug("START ...")
    fig = plot_func(study)
    fig.update_layout(width=1200, height=1200)  # Optionally adjust the plot size
    fig.write_image(filename)
    logger_setup.logger.debug("... FINISH")

def analyse_optuna_study(study):
    logger_setup.logger.debug("START ...")
    # Generate and save the native Optuna plots
    save_plot(plot_optimization_history, study, f'{PATH_OUT_VISUALS}optimization_history_{MODEL_VERSION}.png')
    save_plot(plot_parallel_coordinate, study, f'{PATH_OUT_VISUALS}parallel_coordinate_{MODEL_VERSION}.png')
    save_plot(plot_slice, study, f'{PATH_OUT_VISUALS}slice_{MODEL_VERSION}.png')
    save_plot(plot_contour, study, f'{PATH_OUT_VISUALS}contour_{MODEL_VERSION}.png')
    save_plot(plot_param_importances, study, f'{PATH_OUT_VISUALS}param_importances_{MODEL_VERSION}.png')
    save_plot(plot_edf, study, f'{PATH_OUT_VISUALS}edf_{MODEL_VERSION}.png')
    study_metrics = study.trials_dataframe()
    # retrieve all performance values for each model types studied
    grouped_metrics = study_metrics.groupby('params_model')
    # Apply a function to create a dictionary for each group
    result_dict = grouped_metrics.apply(lambda group: {
        'optuna_trial_number': group['number'].tolist(),
        'optuna_objective_value': group['value'].tolist()
    }).to_dict()
    df_xgb = pd.DataFrame(result_dict.get('xgb'))
    df_rfc = pd.DataFrame(result_dict.get('rfc'))
    utility.plot_line([df_xgb, df_rfc], ['xgb', 'rfc'], ['optuna_trial_number'], ['optuna_objective_value'])
    logger_setup.logger.debug("... FINISH")

def save_artefacts(study, best_models_dict):
    logger_setup.logger.debug("START ...")
    study_full_metrics = study.trials_dataframe()
    # save the metrics to a file
    study_full_metrics.to_csv(f'{PATH_OUT_MODELS}optuna_study_stats_{MODEL_VERSION}.txt')

    for key, value in best_models_dict.items():
        # save the model to a file
        joblib.dump(value, f'{PATH_OUT_MODELS}best_model_pipe_{key}_{MODEL_VERSION}.pkl')
        logger_setup.logger.info(f'Saved best {key} model object as file: {PATH_OUT_MODELS}best_model_pipe_{key}_{MODEL_VERSION}.pkl')
    logger_setup.logger.debug("... FINISH")

def get_feature_names(best_models_dict, orig_train_data_cols):
    logger_setup.logger.debug("START ...")
    feature_names = []
    column_names = []
    for key, value in best_models_dict.items():
        # get the preprocessor part of the pipeleine object
        logger_setup.logger.info(f'Looping through the {key} model object.')
        preproc = value.named_steps['preprocessing']
        for i, (name, trans, column) in enumerate(preproc.transformers_):
            #print(f'Transformer#{i + 1} name is:{beautify(str(name))}')
            logger_setup.logger.debug(f'Transformer#{i + 1} name is:{name}')
            if type(trans) is Pipeline:
                logger_setup.logger.debug('\t Is a Pipeline')
                trans = trans.steps[-1][1]
            else:
                logger_setup.logger.debug('\t Isn\'t a Pipeline')
            if hasattr(trans, 'get_feature_names_out'):
                logger_setup.logger.debug('\t Has get_feature_names_out')
                tmp_feature_names = trans.get_feature_names_out(column)
                feature_names.extend(tmp_feature_names)
                column_names.extend(column)
                logger_setup.logger.debug(f'\t Transformer input = {len(column)} and output = {len(tmp_feature_names)}')
            elif hasattr(trans, 'get_feature_names'):
                logger_setup.logger.debug('\t Has get_feature_names')
                tmp_feature_names = trans.get_feature_names(column)
                feature_names.extend(tmp_feature_names)
                column_names.extend(column)
                logger_setup.logger.debug(f'\t Transformer input = {len(column)} and output = {len(tmp_feature_names)}')
            else:
                logger_setup.logger.debug('\t Doesn\'t have get_feature_names or get_feature_names_out')
                if name == 'remainder' and trans == 'passthrough':
                    logger_setup.logger.debug('\t > It\'s remainder passthrough')
                    tmp_remainder_names = set(orig_train_data_cols) - set(column_names)
                    feature_names.extend(tmp_remainder_names)
                    column_names.extend(column)
                    logger_setup.logger.debug(f'\t Transformer input = {len(column)} and output = {len(column)}')
                else:
                    logger_setup.logger.debug('\t > Not a remainder passthrough')
                    tmp_feature_names = column
                    feature_names.extend(tmp_feature_names)
                    column_names.extend(column)
                    logger_setup.logger.debug(f'\t Transformer input = {len(column)} and output = {len(tmp_feature_names)}')
    logger_setup.logger.info(f'\nThe total feature space has: {len(feature_names)} features. Their names being:\n{feature_names}')
    logger_setup.logger.debug("... FINISH")
    return feature_names
