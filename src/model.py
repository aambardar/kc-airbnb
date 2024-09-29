import numpy as np
import optuna
from utility import beautify
from conf.config import OPTUNA_TRIAL_COUNT, RANDOM_STATE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import logging
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

X_train = None
y_train = None
c_trans = None
logger = logging.getLogger()

def ndcg_at_5(y_true, y_pred_proba, k=5):
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
    return ndcg

def run_hyperparam_tuning(X_train, y_train, col_trans):
    def optuna_objective(trial):
        models = {}
        classifier_models = trial.suggest_categorical('model', ['xgb'])
        print(f'Trial: {trial}')
        print(f'Trial number: {trial.number}')

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
        print(f'Trial {beautify(str(trial.number))} Scoring Starts...')
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
        print("NDCG@5 scores:", scores)
        print("Mean NDCG@5 score:", score)

        print(f'Trial {beautify(str(trial.number))} Before Saving to Models...')
        models[trial.number] = pipe_model
        print(f'Trial {beautify(str(trial.number))} After Saving to Models...')
        return score

    # creation of Optuna study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE))
    X_train = X_train
    y_train = y_train
    c_trans = col_trans
    # optimise the study
    study.optimize(optuna_objective, n_trials=OPTUNA_TRIAL_COUNT)
    return study
