import random
random.seed(42)
import inspect
import numpy as np
import importlib
import pandas as pd
from scipy.spatial.distance import euclidean

IGNORE = [
    'delta_TNLPM',
    'delta_NL', 'delta_ThrowStatement',
    'delta_SwitchStatementCase', 'delta_BreakStatement', 'delta_WhileStatement',
    "delta_Statement", 'delta_ContinueStatement', 'delta_SwitchStatement',
    'delta_NLE'
]
euclidean_similarity = lambda tp, n_sample: [euclidean(row.values, tp) for _, row in n_sample.iterrows()]
valu_diff_distance = lambda tp, n_sample: [n_sample[i] - tp[i] for i in tp if tp[i] != n_sample[i]]


def check_valid(value_diff_key, features_affect_key):
    """
    Checking the correctness of whether the semantic change can be applied.

    :param value_diff_key:
    :type value_diff_key: the number value that need change
    :param features_affect_key:
    :type features_affect_key: The numerical value that the semantic change changes

    :return: Returns True if the semantic change can be performed. Otherwise returns False.
    :rtype: bool
    """
    if value_diff_key >= 0 and 0 <= features_affect_key <= value_diff_key:
        return True
    elif value_diff_key <= 0 and 0 >= features_affect_key >= value_diff_key:
        return True
    return False


def apply_rules(value_diff, tp_sample, ignore=False):
    """
    A function that scans all the semantic changes and executes them if they satisfy the preconditions.

    :param value_diff: Dict with the value need change from tp to n
    :type value_diff: Dict
    :param tp_sample: an instance that classified as TP
    :type tp_sample: list
    :param ignore: Indicates whether to ignore some of the features
    :type ignore: bool

    :return: success, new TP (or None if the success is False)
    :rtype: bool, list
    """
    classes = inspect.getmembers(importlib.import_module("Play.rules"), inspect.isclass)
    dict_apply_rule = {}
    for name, rule_cls in classes:
        can_change = True
        if name not in ['ABC', "Move", 'AddSwitchStatement']:
            features_affect = rule_cls.affect_feature(rule_cls)
            minimal_abs = np.inf
            ignore_time = 0
            for key in features_affect:
                if ignore and key in IGNORE and not all([i in IGNORE for i in list(features_affect.keys())]):
                    ignore_time += 1
                    continue
                elif key in value_diff.keys() and check_valid(value_diff[key], features_affect[key]):
                    if minimal_abs > abs(value_diff[key]) / abs(features_affect[key]):
                        minimal_abs = abs(value_diff[key]) / abs(features_affect[key])
                    continue
                else:
                    can_change = False
                    break
            if can_change and rule_cls.can_apply(rule_cls, tp_sample):
                dict_apply_rule[name] = [rule_cls, -minimal_abs, ignore_time]
    if dict_apply_rule != {}:
        for key in dict_apply_rule:
            for i in range(int(-dict_apply_rule[key][1])):
                rule_cls = dict_apply_rule[key][0]
                yes = False
                yes += rule_cls.apply_if_can(rule_cls, tp_sample)
        return True, dict_apply_rule.keys()
    return False, None


def move(tp_sample, negative_samples):
    """
    Semantically Equivalent Code Generation Semantically Equivalent Code Generation

    :param tp_sample: instance that classified as TP
    :type tp_sample: list
   :param negative_samples: list with instances that classified as N (TN AND FN)
   :type negative_samples: list

    """
    can_change = True
    while can_change:
        index = 0
        negative_without_parent = negative_samples.drop(
            [col for col in negative_samples.columns if "parent" in col], axis=1)
        tp_without_parent = {i: dict(tp_sample)[i] for i in dict(tp_sample) if "parent" not in i}
        distance = euclidean_similarity(list(tp_without_parent.values()), negative_without_parent)
        sorted_ = sorted(range(len(distance)), key=lambda k: distance[k])
        success = False
        while index != len(distance):
            current_n = negative_samples.iloc[sorted_[index]]
            current_n_without_parent = {i: current_n[i] for i in current_n.keys() if 'parent' not in i}
            value_diff = {i: current_n_without_parent[i] - tp_without_parent[i] for i in tp_without_parent if
                          tp_without_parent[i] != current_n_without_parent[i]}
            success = apply_rules(value_diff, tp_sample, ignore=True)
            if success[0]:
                break
            index += 1
        if not success[0]:
            can_change = False


def get_similarities(tp_sample, negative_samples):
    copy_tp = tp_sample.copy()
    move(tp_sample, negative_samples)
    return tp_sample if not all(tp_sample == copy_tp) else None


def play_game(X_test, y_true, y_pred, name_project, model=None, path=None):
    """
    The main algorithm of the process ECG.

    :param X_test: Test vector (Test set from the data)
    :type X_test: DataFrame
    :param y_true:  1d array-like, or label indicator array / sparse matrix Ground truth (correct) target values.
    :type y_true: array
    :param y_pred:  1d array-like, or label indicator array / sparse matrix Estimated targets as returned by a classifier.
    :type y_pred: array
    :param name_project: path to save file
    :type name_project: str

    :return: X_semantics_change
    :rtype: DataFrame
    """
    def find_negative_example(X_test, y_true, y_pred):
        negative_classify = []
        for (_, x), (_, t), p in zip(X_test.iterrows(), y_true.iteritems(), y_pred):
            if p == 0:  # find all negative instance
                negative_classify.append(x.copy().to_list())
        return pd.DataFrame(negative_classify, columns=X_test.columns)

    negative_classify = find_negative_example(X_test, y_true, y_pred)
    X_play = []
    new_y_true = []
    index_success = 0
    index = 0
    index_of_tp, list_index_of_tp, list_index_of_tp_success = -1, [], []
    change_example = []
    for (_, x), (_, t), p in zip(X_test.iterrows(), y_true.iteritems(), y_pred):
        index_of_tp += 1
        if t == 1 and p == 1:  # find all True positive instance
            index += 1
            results = get_similarities(x, negative_classify)
            list_index_of_tp.append(index_of_tp)
            if results is not None:
                X_play.append(list(results))
                change_example.append(list(results))
                new_y_true.append(t)
                index_success += 1
            else:  # can't move
                X_play.append(x.copy().to_list())
                new_y_true.append(t)
        else:
            X_play.append(x.copy().to_list())
            new_y_true.append(t)
    print(f"For {name_project} get {index} example change {index_success}")
    return pd.DataFrame([list(i) for i in X_play], columns=X_test.columns), new_y_true

