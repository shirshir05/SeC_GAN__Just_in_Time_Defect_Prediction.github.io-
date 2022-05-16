import csv
import json
import os
import sys
import numpy as np
import tensorflow
import torch
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.nn import Linear, Dropout, Sequential, BCELoss
from xgboost import XGBClassifier

sys.path.insert(0, '/home/shir0/GAN_VS_RF')
import variable
import Algorithms.Evaluate as Evaluate
from Play.play_algorithm import play_game

tuning_rf_dict = {'cayenne': [50, 'gini', 'auto'], 'kylin': [50, 'gini', 'auto'], 'jspwiki': [50, 'gini', 'auto'],
                  'manifoldcf': [50, 'gini', 'auto'],'manifoldcf_G': [50, 'gini', 'auto'],
                  'commons-lang': [50, 'gini', 'auto'], 'jspwiki_G': [50, 'gini', 'auto'],
                  'tika': [50, 'gini', 'auto'], 'kafka': [100, 'gini', 'auto'], 'kafka_G': [100, 'gini', 'auto'],
                  'zookeeper': [800, 'gini', 'auto'],
                  'zeppelin': [2000, 'gini', 'auto'], 'shiro': [50, 'gini', 'auto'],
                  'logging-log4j2': [50, 'gini', 'auto'], 'activemq-artemis': [50, 'gini', 'auto'],
                  'shindig': [500, 'gini', 'auto'], 'directory-studio': [50, 'gini', 'auto'],
                  'tapestry-5': [500, 'gini', 'auto'], 'openjpa': [50, 'gini', 'auto'], 'knox': [100, 'gini', 'auto'],
                  'commons-configuration': [1000, 'gini', 'auto'], 'xmlgraphics-batik': [50, 'gini', 'auto'],
                  'mahout': [50, 'gini', 'auto'],  'mahout_G': [50, 'gini', 'auto'],
                  'deltaspike': [50, 'gini', 'auto'],
                  'openwebbeans': [100, 'gini', 'auto'], 'commons-collections': [50, 'gini', 'auto'],
                  'cayenne_G': [50, 'gini', 'auto'],
                  'kylin_G': [50, 'gini', 'auto'], 'commons-lang_G': [50, 'gini', 'auto'],
                  'tika_G': [50, 'gini', 'auto'], 'zookeeper_G': [800, 'gini', 'auto'],
                  'zeppelin_G': [2000, 'gini', 'auto'], 'shiro_G': [50, 'gini', 'auto'],
                  'logging-log4j2_G': [50, 'gini', 'auto'], 'activemq-artemis_G': [50, 'gini', 'auto'],
                  'shindig_G': [500, 'gini', 'auto'], 'directory-studio_G': [50, 'gini', 'auto'],
                  'tapestry-5_G': [500, 'gini', 'auto'], 'openjpa_G': [50, 'gini', 'auto'],
                  'knox_G': [100, 'gini', 'auto'], 'commons-configuration_G': [1000, 'gini', 'auto'],
                  'xmlgraphics-batik_G': [50, 'gini', 'auto'], 'deltaspike_G': [50, 'gini', 'auto'],
                  'openwebbeans_G': [100, 'gini', 'auto'], 'commons-collections_G': [50, 'gini', 'auto']}

# {'cayenne': ['gini', 2, 'auto', None, 0.1, 0, True],
#               'cayenne_G': ['gini', 2, 'auto', None, 0.1, 0, True],
#               'kylin': ['gini', 2, 25, None, 0.1, 0, True],
#               'kylin_G': ['gini', 2, 25, None, 0.1, 0, True],
#               'jspwiki': ['gini', 2, 5, None, 0.1, 0, True],
#               'jspwiki_G': ['gini', 2, 5, None, 0.1, 0, True],
#               'manifoldcf': ['entropy', 2, None, 5, 0.1, 0, True],
#               'manifoldcf_G': ['entropy', 2, None, 5, 0.1, 0, True],
#               'commons-lang': ['gini', 50, None, None, 0.1, 0, True],
#               'commons-lang_G': ['gini', 50, None, None, 0.1, 0, True],
#               'tika': ['gini', 2, 50, 7, 0.1, 0, True],
#               'tika_G': ['gini', 2, 50, 7, 0.1, 0, True],
#               'kafka': ['entropy', 2, None, 7, 0.1, 0, True],
#               'kafka_G': ['entropy', 2, None, 7, 0.1, 0, True],
#               'zookeeper': ['gini', 3, None, 7, 0.1, 0, False],
#               'zookeeper_G': ['gini', 3, None, 7, 0.1, 0, False],
#               'zeppelin': ['gini', 5, None, None, 0.1, 0, False],
#               'zeppelin_G': ['gini', 5, None, None, 0.1, 0, False],
#               'shiro': ['gini', 2, 3, None, 0.1, 0, False],
#               'shiro_G': ['gini', 2, 3, None, 0.1, 0, False],
#               'logging-log4j2': ['gini', 2, 50, None, 0.1, 0, False],
#               'logging-log4j2_G': ['gini', 2, 50, None, 0.1, 0, False],
#               'activemq-artemis': ['gini', 2, 50, 7, 0.1, 0, False],
#               'activemq-artemis_G': ['gini', 2, 50, 7, 0.1, 0, False],
#               'shindig': ['gini', 2, 5, None, 0.1, 0, False],
#               'directory-studio': ['entropy', 5, None, 3, 0.1, 0, False],
#               'tapestry-5': ['entropy', 2, 50, 3, 0.1, 0, False],
#               'shindig_G': ['gini', 2, 5, None, 0.1, 0, False],
#               'directory-studio_G': ['entropy', 5, None, 3, 0.1, 0, False],
#               'tapestry-5_G': ['entropy', 2, 50, 3, 0.1, 0, False],
#               'openjpa': ['gini', 2, 50, None, 0.1, 0, False], 'knox': ['gini', 3, None, None, 0.1, 0, False],
#               'commons-configuration': ['gini', 2, 'log2', 5, 0.1, 0, False],
#               'xmlgraphics-batik': ['gini', 10, 50, None, 0.1, 0, False],
#               'mahout': ['gini', 2, 'auto', 5, 0.1, 0, False], 'deltaspike': ['gini', 3, 25, None, 0.1, 0, False],
#               'openjpa_G': ['gini', 2, 50, None, 0.1, 0, False], 'knox_G': ['gini', 3, None, None, 0.1, 0, False],
#               'commons-configuration_G': ['gini', 2, 'log2', 5, 0.1, 0, False],
#               'xmlgraphics-batik_G': ['gini', 10, 50, None, 0.1, 0, False],
#               'mahout_G': ['gini', 2, 'auto', 5, 0.1, 0, False],
#               'deltaspike_G': ['gini', 3, 25, None, 0.1, 0, False],
#               'openwebbeans_G': ['entropy', 2, 50, None, 0.1, 0, False],
#               'commons-collections_G': ['gini', 2, None, None, 0.1, 0, False],
#               'openwebbeans': ['entropy', 2, 50, None, 0.1, 0, False],
#               'commons-collections': ['gini', 2, None, None, 0.1, 0, False]
#               }

# n_estimators_old =

tuning_lr_dict = {'cayenne': ['l2', 'liblinear', 0.001], 'kylin': ['l2', 'saga', 0.001],
                  'jspwiki': ['l2', 'newton-cg', 0.001], 'manifoldcf': ['l1', 'saga', 0.001],
                  'commons-lang': ['l2', 'liblinear', 0.001], 'tika': ['l2', 'lbfgs', 0.001],
                  'kafka': ['l1', 'saga', 0.001], 'zookeeper': ['elasticnet', 'saga', 0.001],
                  'zeppelin': ['l2', 'liblinear', 0.001], 'shiro': ['l2', 'saga', 0.001],
                  'logging-log4j2': ['l1', 'liblinear', 0.001], 'activemq-artemis': ['l1', 'liblinear', 0.001],
                  'shindig': ['l2', 'saga', 0.001], 'directory-studio': ['l2', 'newton-cg', 0.001],
                  'tapestry-5': ['l1', 'liblinear', 0.001], 'openjpa': ['l2', 'sag', 0.5],
                  'knox': ['l2', 'saga', 0.001], 'commons-configuration': ['l2', 'liblinear', 0.001],
                  'xmlgraphics-batik': ['l2', 'saga', 0.001], 'mahout': ['l1', 'saga', 0.001],
                  'deltaspike': ['l1', 'saga', 0.001], 'openwebbeans': ['l2', 'liblinear', 0.001],
                  'commons-collections': ['l2', 'saga', 0.001], 'kylin_G': ['l2', 'saga', 0.001],
                  'commons-lang_G': ['l2', 'liblinear', 0.001], 'tika_G': ['l2', 'lbfgs', 0.001],
                  'zookeeper_G': ['l2', 'saga', 0.001], 'zeppelin_G': ['l2', 'liblinear', 0.001],
                  'shiro_G': ['l2', 'saga', 0.001], 'logging-log4j2_G': ['l1', 'liblinear', 0.001],
                  'activemq-artemis_G': ['l1', 'liblinear', 0.001], 'shindig_G': ['l2', 'saga', 0.001],
                  'directory-studio_G': ['l2', 'newton-cg', 0.001], 'tapestry-5_G': ['l1', 'liblinear', 0.001],
                  'openjpa_G': ['l2', 'sag', 0.3], 'knox_G': ['l2', 'saga', 0.001],
                  'commons-configuration_G': ['l2', 'liblinear', 0.001], 'xmlgraphics-batik_G': ['l2', 'saga', 0.001],
                  'deltaspike_G': ['l1', 'saga', 0.001], 'openwebbeans_G': ['l2', 'liblinear', 0.001],
                  'commons-collections_G': ['l2', 'saga', 0.001], 'cayenne_G': ['l2', 'liblinear', 0.001],
                  'jspwiki_G': ['l2', 'newton-cg', 0.001], 'manifoldcf_G': ['l1', 'saga', 0.001],
                  'kafka_G': ['l1', 'saga', 0.001], 'mahout_G': ['l1', 'saga', 0.001]}
tuning_xgb_dict = {'cayenne': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'cayenne_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'kylin': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'jspwiki': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'manifoldcf': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'commons-lang': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'tika': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'kafka': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   "zookeeper": ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'zeppelin': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'shiro': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'logging-log4j2': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'activemq-artemis': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'shindig': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'directory-studio': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'tapestry-5': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'openjpa': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'knox': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'commons-configuration': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'xmlgraphics-batik': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'mahout': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'deltaspike': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'openwebbeans': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'commons-collections': ['deviance', 0.01, 500, 'friedman_mse', 2],

                   'kylin_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'jspwiki_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'manifoldcf_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'commons-lang_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'tika_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'kafka_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   "zookeeper_G": ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'zeppelin_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'shiro_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'logging-log4j2_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'activemq-artemis_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'shindig_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'directory-studio_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'tapestry-5_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'openjpa_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'knox_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'commons-configuration_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'xmlgraphics-batik_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'mahout_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'deltaspike_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'openwebbeans_G': ['deviance', 0.01, 500, 'friedman_mse', 2],
                   'commons-collections_G': ['deviance', 0.01, 500, 'friedman_mse', 2],

                   }

tuning_nb_dict = {'cayenne': [0.657933224657568], 'kylin': [0.2848035868435802], 'jspwiki': [0.15199110829529336],
                  'manifoldcf': [0.0533669923120631], 'commons-lang': [0.001873817422860383],
                  'tika': [0.01873817422860384], 'kafka': [0.004328761281083057], 'zookeeper': [0.02848035868435802],
                  'zeppelin': [0.1], 'shiro': [0.3511191734215131], 'logging-log4j2': [0.004328761281083057],
                  'activemq-artemis': [0.0003511191734215131], 'shindig': [0.2848035868435802],
                  'directory-studio': [0.03511191734215131], 'tapestry-5': [0.0015199110829529332],
                  'openjpa': [1.232846739442066e-06], 'knox': [0.012328467394420659],
                  'commons-configuration': [0.0015199110829529332], 'xmlgraphics-batik': [0.004328761281083057],
                  'mahout': [0.1873817422860384], 'deltaspike': [0.533669923120631],
                  'openwebbeans': [0.02310129700083159], 'commons-collections': [0.0023101297000831605]}

space_lr = {'penalty': ['l1', 'l2', 'elasticnet'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            "l1_ratio": [0.001, 0.01, 0.1, 0.5, 0.3]}

space_xgb = {
    "loss": ['deviance', 'exponential'],
    'learning_rate': [0.01],  # 0.0001, 0.001, , 0.1, 0.2, 0.3
    'n_estimators': [500],  # 25, 50, 100,  , 1000
    'criterion': ['friedman_mse'],  # , 'squared_error', 'mse', 'mae'
    "min_samples_split": [2]  # , 5, 10, 0.1
}

space_rf = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [2, 3, 5, 10, 25, 50, 100, 1000],
    'max_features': [3, 5, 10, 25, 50, 'auto', 'log2', None],
    'max_depth': [None, 3, 5, 7, 10, 100, 1000],
    'min_samples_split': [0.1],  # 2, 5, 10,
    'min_impurity_decrease': [0],  # 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
    'warm_start': [False]
    # "n_estimators": [50,  100, 500, 800, 1000, 1500, 2000],  # 50,
    # "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2"]
    # "criterion": ["gini"], "max_features": ["auto"]
}

space_nb = {'var_smoothing': np.logspace(0, -9, num=100)}

tuning_D = {"D":
                {"CTGAN": False, "number_commit_without_bug": 0.8, "epochs": 200, "batch_size": 30,
                 "batch_size_G": 0, "discriminator_steps": 500, "layers": [128, 256], "lr_D": 0.002,
                 "lr_G": 0},
            "D_G": {
                "CTGAN": True, "number_commit_without_bug": 0.8, "epochs": 200, "batch_size": 30,
                "batch_size_G": 30, "discriminator_steps": 500, "layers": [128, 256], "lr_D": 0.002,
                "lr_G": 0.02}}


def init_ctgna(two_g=True):
    """
    The function reads the file containing the faked instances.

    :param two_g: A parameter that specifies whether to use information from one G or two G.
    A default value is True and therefore two G are used.
    :type bool
    :return: data_from_G (X) , y_from_G (Y)
    :rtype: DataFrame, Series

    .. note::
            You must create the fake data using the generator from file Generate_fake.py

    """
    if two_g:
        data = pd.read_csv(os.path.join(NAME_PROJECT, "train_test", f"new_fake_data_{p}_bug_2000.csv"))  # _1000_512
        data = data.iloc[:, 1:]
        data['commit insert bug?'] = 1
        data_nbug = pd.read_csv(os.path.join(NAME_PROJECT, "train_test", f"new_fake_data_{p}_nbug.csv"))  # _500_512
        data_nbug = data_nbug.iloc[:, 1:]
        data_nbug['commit insert bug?'] = 0
        data = pd.concat([data_nbug, data])
    else:
        data = pd.read_csv(os.path.join(NAME_PROJECT, "train_test", f"new_fake_data_{p}_all_1000_512.csv"))
        data = data.iloc[:, 1:]

    y_from_G = data.pop('commit insert bug?')
    data_from_G = data
    return data_from_G, y_from_G


def read_data():
    """
    The function reads the data after it has completed the preprocessing process.

    :return: X_train, X_test, X_valid, y_train, y_test, y_valid
    :rtype: DataFrame,, DataFrame, DataFrame, Series, Series, Series

    .. note:: You must create the train, validation and test set using from file main_create_data.py
    """

    def read_help(path):
        df = pd.read_csv(NAME_PROJECT + '/train_test/' + path)
        df = df.iloc[:, 1:]
        y_train = df.pop('commit insert bug?')
        return df, y_train

    X_train, y_train = read_help(f"train_{p}.csv")
    X_test, y_test = read_help(f"test_{p}.csv")
    X_valid, y_valid = read_help(f"val_{p}.csv")
    features_check = [col for col in X_train.columns if col in variable.features_check_before_pre_process]
    X_train = X_train[features_check]
    X_test = X_test[features_check]
    X_valid = X_valid[features_check]
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def init_value():
    """
    Create a folder to save the results.
    """
    if not os.path.exists(os.path.join(NAME_PROJECT, "Results", name_model)):
        os.mkdir(os.path.join(NAME_PROJECT, "Results", name_model))
    print(f"Project {NAME_PROJECT}")


def tuning(x_val, y_val, space, estimator):
    """
    Function that performs a tuning process to the estimator on the validation set.

    :param x_val: Training vector (Validation set from the data)
    :type x_val: DataFrame
    :param y_val: Target relative to X for classification
    :type
    y_val: Series
    :param space: Dictionary with parameters names (str) as keys and lists of parameter settings to try
    as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are
    explored. This enables searching over any sequence of parameter settings.
    :type space : Dictionary(str,
    list)
    :param estimator: Classifier - this is assumed to implement the scikit-learn estimator interface.
    :type estimator: object
    :return: Parameter setting that gave the best results on the hold out data.
    :rtype: Dictionary

    """
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    search = GridSearchCV(estimator, space, scoring='f1', n_jobs=-1, cv=cv_inner, refit=True)
    search.fit(x_val, y_val)
    return search.best_params_


def train(X_train, y_train, X_test, y_test, model):
    """
    The main function that performs the training process. The function extracts the evaluation metrics after the
    training process on the test set. In addition, the function executes the ECG algorithm and evaluates the model on
    X_semantic_change.
    The function also calculates the importances of the features and saves them in the results folder in a file named
    importances_path.

    :param X_train: Training vector (Training set from the data)
    :type X_train: DataFrame
    :param y_train: Target relative to X_train
    :type y_train: Series
    :param X_test: Test vector (Test set from the data)
    :type X_test: DataFrame
    :param y_test: Target relative to X_test
    :type y_test: Series
    :param model: Classifier - this is assumed to implement the scikit-learn estimator interface.
    :type model: Union[sklearn.ensemble.RandomForestClassifier, sklearn.linear_model.LogisticRegression]

    :return: scores (on X test), play_scores (on X_semantic_change)
    :rtype: list, list
    """
    model.fit(X_train, y_train)
    # if "RF" in name_model and with_G:
    #     model.n_estimators += 100
    #     model.fit(data_from_G, y_from_G)
    scores, play_scores, importances = [], [], []
    if scaler is not None:
        X_test_saclar = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        predictions_proba = list(zip(*model.predict_proba(X_test_saclar)))
        predictions = model.predict(X_test_saclar)
    else:
        predictions_proba = list(zip(*model.predict_proba(X_test)))
        predictions = model.predict(X_test)
    scores.append(Evaluate.evaluate_on_test(y_test, predictions, [0, 1], predictions_proba))
    try:  # RF
        feature_importance = abs(model.feature_importances_)
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        importances.append(dict(zip(X_train.columns, feature_importance)))
    except:  # LF
        try:
            feature_importance = abs(model.coef_[0])
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            importances.append(
                dict(zip(X_train.columns, feature_importance)))
        except:  # NB
            imps = permutation_importance(model, X_train, y_train)
            feature_importance = imps.importances_mean
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            importances.append(
                dict(zip(X_train.columns, feature_importance)))
    with open(os.path.join(NAME_PROJECT, "Results", name_model, f"importances_{path}.json"), 'w') as file:
        json.dump(str(importances), file)
    X_play, new_y_true = play_game(X_test, y_test, predictions, NAME_PROJECT, model=model, path=path)
    if scaler is not None:
        X_play_saclar = pd.DataFrame(scaler.transform(X_play), columns=X_test.columns)
        predictions_proba = list(zip(*model.predict_proba(X_play_saclar)))
        predictions = model.predict(X_play_saclar)
    else:
        predictions_proba = list(zip(*model.predict_proba(X_play)))
        predictions = model.predict(X_play)
    play_scores.append(Evaluate.evaluate_on_test(new_y_true, predictions, [0, 1], predictions_proba))
    write_results(scores, play_scores, importances)
    return scores, play_scores


def write_results(scores, play_scores, importances=[]):
    """
    Write the results to a file.
    :param scores:
    :type scores: list
    :param play_scores:
    :type play_scores: list
    :param importances:
    :type importances: list
    """
    with open(os.path.join(NAME_PROJECT, "Results", name_model, f"play_scores_{project}_{path}.json"), 'w') as file:
        json.dump(play_scores, file)
    with open(os.path.join(NAME_PROJECT, "Results", name_model, f"scores_{project}_{path}.json"), 'w') as file:
        json.dump(scores, file)
    if importances is not []:
        with open(os.path.join(NAME_PROJECT, "Results", name_model, f"importances_{project}_{path}.json"), 'w') as file:
            json.dump(str(importances), file)


# region FCNN
class Discriminator(nn.Module):
    """
    The class represents a neural network.
    """

    def __init__(self, commit_shape):
        """
        A constructor that initializes the neural network.
        initializes is performed according to the parameters in the dictionary tuning_D.
        :param commit_shape: Number of columns in data.
        :type commit_shape: int
        :rtype: object
        """
        super(Discriminator, self).__init__()
        self.commit_shape = commit_shape
        layers = tuning_D[name_G]["layers"]
        seq = []
        dropout_rate = 0.3
        layers = [commit_shape] + layers
        for i in range(len(layers) - 1):
            seq += [Linear(layers[i], layers[i + 1])]
            seq += [nn.Tanh(), Dropout(dropout_rate)]
        seq += [Linear(layers[len(layers) - 1], 1)]
        seq += [nn.Sigmoid()]
        self.model_seq = Sequential(*seq)

    def forward(self, input):
        """
        The forward process - defines the computation performed at every call.

        :param input:
        :type input:
        :return: model_output
        """
        model_output = self.model_seq(input)
        return model_output


def generate_batch_real():
    """
    .. generate_batch_real:

    Random selection of samples from the train set for a single batch.
    The batch size is determined by the keyword "batch_size" in the dictionary tuning_D (default 30).. The percentage of instances
    that did not induce a defect is defined by the keyword "number_commit_without_bug" (default 0.8).

    :return: [X_real, labels]
    :rtype: DataFrame, Series
    """
    batch_size = tuning_D[name_G]['batch_size']
    number_commit_without_bug = tuning_D[name_G]['number_commit_without_bug']
    idx_zero = np.random.randint(0, x_zero_only.shape[0],
                                 int(batch_size * number_commit_without_bug))
    idx_one = np.random.randint(0, x_one_only.shape[0],
                                batch_size - int(batch_size * number_commit_without_bug))
    X_real, labels = np.concatenate((x_one_only[idx_one], x_zero_only[idx_zero])), \
                     np.concatenate((y_one_only[idx_one], y_zero_only[idx_zero]))
    idxs = np.arange(0, X_real.shape[0])
    np.random.shuffle(idxs)
    X_real = X_real[idxs]
    labels = labels[idxs]
    return [X_real, labels]


def train_iteration_D_real_data(write=False):
    """
    The function trains the model on batch of real example and calculates the loss using BCELoos.
    This function use in :ref:`generate_batch_real` to generate data for ths batch.

    :param write: Indicates whether to write the loss result to the file
    :type write: bool
    """
    data_batch = generate_batch_real()
    predict_real = model(torch.tensor(data_batch[0][:, 0:, 0]).float())
    loss = BCELoss()
    real_loss = loss(predict_real[:, 0].float(), torch.tensor(data_batch[1])[:, 0].float())
    D_optimizer.zero_grad()
    real_loss.backward(retain_graph=True)
    D_optimizer.step()
    if write:
        with open(os.path.join(NAME_PROJECT, "Results", "LOSS", path + "_real.csv"), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                [str(real_loss.data).split("(")[1].split(")")[0]])


def generate_batch_fake():
    """
    .. generate_batch_fake:

    Random selection of samples from the fake data set for a single batch.
    The batch size is determined by the keyword "batch_size" in the dictionary tuning_D (default 30).
    The percentage of instances that did not induce a defect is defined by the keyword "number_commit_without_bug" (default 0.8).

    :return: [X_fake, labels]
    :rtype: DataFrame, Series
    """
    batch_size_G = tuning_D[name_G]['batch_size_G']
    number_commit_without_bug = tuning_D[name_G]['number_commit_without_bug']
    idx_zero = np.random.randint(0, x_zero_only_CTGAN.shape[0],
                                 int(batch_size_G * number_commit_without_bug))
    idx_one = np.random.randint(0, x_one_only_CTGAN.shape[0],
                                batch_size_G - int(batch_size_G * number_commit_without_bug))
    X_fake, labels = np.concatenate((x_one_only_CTGAN[idx_one], x_zero_only_CTGAN[idx_zero])), \
                     np.concatenate((y_one_only_CTGAN[idx_one], y_zero_only_CTGAN[idx_zero]))
    idxs = np.arange(0, X_fake.shape[0])
    np.random.shuffle(idxs)
    X_fake = X_fake[idxs]
    labels = labels[idxs]
    return X_fake[:, :, 0], labels


def train_iteration_D_fake_data(write=False):
    """
    The function trains the model on batch of fake example and calculates the loss using BCELoos.
    This function use in :ref:`generate_batch_fake` to generate data for ths batch.

    :param write: Indicates whether to write the loss result to the file
    :type write: bool
    """
    data_batch = generate_batch_fake()
    predict_real = model(torch.tensor(data_batch[0]).float())
    loss = BCELoss()
    loss_fake = loss(predict_real[:, 0].float(), torch.tensor(data_batch[1])[:, 0].float())
    D_optimizer_fake.zero_grad()
    loss_fake.backward(retain_graph=True)
    D_optimizer_fake.step()
    if write:
        with open(os.path.join(NAME_PROJECT, "Results", "LOSS", path + "_fake.csv"), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                [str(loss_fake.data).split("(")[1].split(")")[0]])


def train_FCNN():
    """
    The main function that performs training of the model.
    The epoch number parameter is stored in the dictionary tuning_D.
    """
    batch_size = tuning_D[name_G]['batch_size']
    steps_per_epoch = max(len(X_train) // batch_size, 1)
    discriminator_steps = tuning_D[name_G]["discriminator_steps"]
    if discriminator_steps != 0:
        steps_per_epoch = max(len(X_train) // (batch_size * discriminator_steps), 1)
    for epoch in range(tuning_D[name_G]['epochs']):
        for iteration in range(steps_per_epoch):
            for n in range(discriminator_steps):
                train_iteration_D_real_data(write=True if iteration == steps_per_epoch - 1 else False)
            train_iteration_D_real_data(write=True if iteration == steps_per_epoch - 1 else False)
            if "G" in name_G:
                train_iteration_D_fake_data(write=True if iteration == steps_per_epoch - 1 else False)


# endregion

if __name__ == '__main__':
    projects = [
        'cayenne',
        'kylin',
        'jspwiki',
        'manifoldcf',
        'commons-lang',
        'tika',
        'kafka',
        'zookeeper', 'zeppelin', 'shiro', 'logging-log4j2',
        'activemq-artemis', 'shindig',
        'directory-studio', 'tapestry-5',
        'openjpa', 'knox',
        'commons-configuration',
        'xmlgraphics-batik', 'mahout',
        'deltaspike',
        'openwebbeans', "commons-collections"

    ]
    path = sys.argv[1]
    name_model = sys.argv[2]
    with_G = bool(int(sys.argv[3]))

    print(path, name_model, with_G)

    for project in projects:
        tensorflow.random.set_seed(420)
        np.random.seed(420)
        torch.manual_seed(420)
        NAME_PROJECT = f"../Data/{project}"
        p = project
        name_G = name_model
        if with_G:
            project += "_G"
            name_G += "_G"
        try:
            init_value()
            X_train, X_test, X_valid, y_train, y_test, y_valid = read_data()
            scaler = None
            if any([True for i in ["LR", "D", "NB", "XGB"] if i in name_model]):
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train), columns=X_test.columns)
                X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_test.columns)
            if with_G:
                data_from_G, y_from_G = init_ctgna()
                if scaler is not None:
                    data_from_G = pd.DataFrame(scaler.transform(data_from_G), columns=X_test.columns)
                if "D" not in name_model:
                    sm = SMOTE(random_state=42)
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                    X_train = pd.concat([X_train, data_from_G])
                    y_train = list(y_train) + [i for i in y_from_G]
                    # if "RF" not in name_model:
                    #     X_train = pd.concat([X_train, data_from_G])
                    #     y_train = list(y_train) + [i for i in y_from_G]
                else:
                    X_CTGAN = np.expand_dims(data_from_G, axis=-1)
                    Y_CTGAN = np.expand_dims(y_from_G, axis=1)
                    x_zero_only_CTGAN = X_CTGAN[np.where(Y_CTGAN == [0])[0]]
                    y_zero_only_CTGAN = Y_CTGAN[np.where(Y_CTGAN == [0])[0]]
                    x_one_only_CTGAN = X_CTGAN[np.where(Y_CTGAN == [1])[0]]
                    y_one_only_CTGAN = Y_CTGAN[np.where(Y_CTGAN == [1])[0]]
            else:
                if "D" not in name_model:
                    sm = SMOTE(random_state=42)
                    X_train, y_train = sm.fit_resample(X_train, y_train)
            if "RF" in name_model:
                if tuning_rf_dict.get(project, 0) == 0:
                    param = tuning(X_valid, y_valid, space_rf, RandomForestClassifier(random_state=42))
                    tuning_rf_dict[project] = [param['criterion'], param['n_estimators'], param['max_features'],
                                               param['max_depth'], param['min_samples_split'],
                                               param['min_impurity_decrease'], param['warm_start']]
                    print(tuning_rf_dict)
                # tuning_rf_dict[project][4] = 2  # min_samples_split (default)
                # tuning_rf_dict[project][2] = 'auto'  # max_features (default)
                # tuning_rf_dict[project][6] = False  # warm_start
                # tuning_rf_dict[project][1] = 500  # n_estimators
                # # tuning_rf_dict[project][3] = 500  # max_depth
                # # tuning_rf_dict[project][0] = 'entropy'
                # tuning_rf_dict[project][2] = 'log2'
                # model = RandomForestClassifier(random_state=42,
                #                                criterion=tuning_rf_dict[project][0],
                #                                n_estimators=tuning_rf_dict[project][1],
                #                                max_features=tuning_rf_dict[project][2],  # default="auto"
                #                                max_depth=tuning_rf_dict[project][3],  # default=None
                #                                min_samples_split=tuning_rf_dict[project][4],  # default=2
                #                                min_impurity_decrease=tuning_rf_dict[project][5],  # default=0.0
                #                                warm_start=tuning_rf_dict[project][6])  # default=False
                # tuning_rf_dict[project][1] = 'entropy'
                model = RandomForestClassifier(random_state=42, n_estimators=tuning_rf_dict[project][0],
                                               criterion=tuning_rf_dict[project][1],
                                               max_features=tuning_rf_dict[project][2])
            elif "LR" in name_model:
                if tuning_lr_dict.get(project, 0) == 0:
                    param = tuning(X_valid, y_valid, space_lr, LogisticRegression())
                    tuning_lr_dict[project] = [param['penalty'], param['solver'], param['l1_ratio']]

                model = LogisticRegression(random_state=42, penalty=tuning_lr_dict[project][0],
                                           solver=tuning_lr_dict[project][1], l1_ratio=tuning_lr_dict[project][2])
            elif "XGB" in name_model:
                if tuning_xgb_dict.get(project, 0) == 0:
                    param = tuning(X_valid, y_valid, space_xgb,
                                   XGBClassifier(verbosity=0, silent=True, use_label_encoder=False))
                    tuning_xgb_dict[project] = [param['loss'], param['learning_rate'], param['n_estimators'],
                                                param['criterion'], param['min_samples_split']]

                tuning_xgb_dict[project][1] = 0.001
                model = XGBClassifier(random_state=42, loss=tuning_xgb_dict[project][0],
                                      learning_rate=tuning_xgb_dict[project][1],
                                      n_estimators=tuning_xgb_dict[project][2], criterion=tuning_xgb_dict[project][3],
                                      min_samples_split=tuning_xgb_dict[project][4], verbosity=0, silent=True,
                                      use_label_encoder=False)
                print(tuning_xgb_dict[project])

            elif "NB" in name_model:
                if tuning_nb_dict.get(project, 0) == 0:
                    param = tuning(X_valid, y_valid, space_nb, GaussianNB())
                    tuning_nb_dict[project] = [param['var_smoothing']]
                model = GaussianNB(var_smoothing=tuning_nb_dict[project][0])
            else:  # D
                X = np.expand_dims(X_train, axis=-1)
                Y = np.expand_dims(y_train, axis=1)
                x_zero_only = X[np.where(Y == [0])[0]]
                y_zero_only = Y[np.where(Y == [0])[0]]
                x_one_only = X[np.where(Y == [1])[0]]
                y_one_only = Y[np.where(Y == [1])[0]]
                model = Discriminator(X_train.shape[1])
                D_optimizer = optim.Adam(model.parameters(), lr=tuning_D[name_G]["lr_D"], betas=(0.5, 0.9),
                                         eps=0.00001,
                                         weight_decay=0.001)
                D_optimizer_fake = optim.Adam(model.parameters(), lr=tuning_D[name_G]["lr_G"], betas=(0.5, 0.9),
                                              eps=0.00001,
                                              weight_decay=0.001)  # 0.001 -> 0.00001
            if "D" not in name_model:
                score, play_scores = train(X_train, y_train, X_test, y_test, model)

            else:
                train_FCNN()
                score, play_scores = Evaluate.evaluate(str(path) + name_G, X_test, scaler, y_test, model, NAME_PROJECT,
                                                       play=True, project=project, CTGAN=None)
            print(score)
            print(play_scores)
        except Exception as e:
            print(e)
            pass
    print(tuning_xgb_dict)
    # print(tuning_rf_dict)
    # print(tuning_lr_dict)
    # print(tuning_nb_dict)
