import json
import os
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from Play import play_algorithm as Play


def evaluate_on_test(y_true, y_pred, classes, proba, tensor=False):
    """
    The function calculates the metrics on the test set.

    :param y_true:  1d array-like, or label indicator array / sparse matrix Ground truth (correct) target values.
    :type y_true: array
    :param y_pred:  1d array-like, or label indicator array / sparse matrix Estimated targets as returned by a classifier.
    :type y_pred: array
    :param classes: [0, 1]
    :type classes: list of classes
    :param proba:1d array-like, or label indicator array / sparse matrix Estimated targets as returned by a classifier.
    :type proba: array
    :param tensor: if use model that return only proba (default False)
    :type tensor: bool

    :return: scores with the metrics (f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, pr_auc_score, tp, fn, tn and fp.
    :rtype: Dict

    """
    def pr_auc_score(y_true, y_score):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
        return metrics.auc(recall, precision)

    scores = {}
    if tensor:
        y_prob_true = proba
    else:
        try:
            y_prob_true = dict(zip(classes, proba))['1']
        except:
            y_prob_true = dict(zip(classes, proba))[1]
    scores['f1_score'] = metrics.f1_score(y_true, y_pred)
    scores['precision_score'] = metrics.precision_score(y_true, y_pred)
    scores['recall_score'] = metrics.recall_score(y_true, y_pred)
    try:
        scores['roc_auc_score'] = metrics.roc_auc_score(y_true, y_prob_true)
    except:
        scores['roc_auc_score'] = metrics.roc_auc_score(y_true, y_prob_true.detach().numpy())
    scores['accuracy_score'] = metrics.accuracy_score(y_true, y_pred)
    scores['pr_auc_score'] = pr_auc_score(y_true, y_pred)
    scores['tn'], scores['fp'], scores['fn'], scores['tp'] = [int(i) for i in
                                                              list(confusion_matrix(y_true, y_pred).ravel())]
    return scores


def evaluate(name, X_test, scaler, y_test, D, NAME_PROJECT, play, project, CTGAN=None,
             threshold=0.5):
    """
    A function that performs the evaluation on a deep learning model.

    :param name: path save file
    :type nams: str
    :param X_test: Test vector (Test set from the data)
    :type X_test: DataFrame
    :param scaler: Standard Scaler object to do the transform
    :type scaler: StandardScaler
    :param y_test: Target relative to X_test
    :type y_test: Series
    :param D: class represents a neural network.
    :type D: nn.Module
    :param NAME_PROJECT: path to save file
    :type NAME_PROJECT: str
    :param play: bool indicate if run ECG process
    :type play: bool
    :param project: name project
    :type project: str
    :param threshold: The threshold define the classes
    :type threshold: double

    :return: scores, scores_play -  Dict with this metrics (f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, pr_auc_score, tp, fn, tn and fp.
    :rtype: Dict, Dict
    """
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if CTGAN:
        X_test_transformer = CTGAN._transformer.transform(X_test)
    else:
        if scaler is not None:
            X_test_scaler = pd.DataFrame(scaler.transform(X_test))
        else:
            X_test_scaler = pd.DataFrame(X_test)
        X_test_transformer = X_test_scaler
    try:
        pred = D.predict(X_test_transformer)
        if isinstance(pred, tuple):
            real_fake, pred = pred
    except:  # for pytorch
        try:
            D.eval()  # for dropout
            with torch.no_grad():
                pred = D(torch.tensor(X_test_transformer).to(device).float())
            if isinstance(pred, tuple):
                real_fake, pred = pred
        except:
            D.eval()
            with torch.no_grad():
                pred = D(torch.tensor(X_test_transformer.values).float())
            if isinstance(pred, tuple):
                real_fake, pred = pred
    if len(pred) == 2:
        predictions_proba = pred[0][:, 0]
    else:
        predictions_proba = pred[:, 0]
    predictions = np.where(predictions_proba.cpu() < threshold, 0, 1)
    scores = evaluate_on_test(y_test, predictions, None, predictions_proba.cpu(), tensor=True)
    with open(os.path.join(NAME_PROJECT, "Results", "D", name + "_train_scores.json"), 'w') as file:
        json.dump(scores, file)
    scores_play = None
    if play:
        # predictions = np.where(predictions_proba.cpu() < threshold, 0, 1)
        X_play, new_y_true = Play.play_game(X_test, y_test, predictions, NAME_PROJECT, D)
        if CTGAN:
            X_play_transformer = CTGAN._transformer.transform(X_play)
        if scaler is not None:
            X_play_transformer = pd.DataFrame(scaler.transform(X_play))
        try:
            pred = D.predict(X_play_transformer.values)
        except:  # for pytorch
            import torch
            try:
                D.eval()
                with torch.no_grad():
                    pred = D(torch.tensor(X_play_transformer.to_numpy()).float())
                if isinstance(pred, tuple):
                    real_fake, pred = pred
            except:
                try:
                    D.eval()
                    with torch.no_grad():
                        pred = D(torch.tensor(X_play_transformer).to(device).float())
                    if isinstance(pred, tuple):
                        real_fake, pred = pred
                except:
                    D.eval()
                    with torch.no_grad():
                        pred = D(torch.tensor(X_play_transformer.values).float())
                    if isinstance(pred, tuple):
                        real_fake, pred = pred
        if len(pred) == 2:
            predictions_proba = pred[0][:, 0]
        else:
            predictions_proba = pred[:, 0]

        predictions = np.where(predictions_proba.cpu() < threshold, 0, 1)
        scores_play = evaluate_on_test(new_y_true, predictions, None, predictions_proba.cpu(), tensor=True)
        with open(os.path.join(NAME_PROJECT, "Results", "D", name + "_train_play_scores.json"), 'w') as file:
            json.dump(scores_play, file)
    return scores, scores_play
