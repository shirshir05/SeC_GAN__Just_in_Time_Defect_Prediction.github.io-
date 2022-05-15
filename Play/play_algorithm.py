import csv
import sys
import random

import os
from sklearn.cluster import KMeans

random.seed(42)
import inspect
import numpy as np
import importlib
import pandas as pd
from scipy.spatial.distance import cityblock
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

sys.path.insert(0, '/home/shir0/GAN_VS_RF')

IGNORE = [
    'delta_TNLPM',
    'delta_NL', 'delta_ThrowStatement',
    'delta_SwitchStatementCase', 'delta_BreakStatement', 'delta_WhileStatement',
    "delta_Statement", 'delta_ContinueStatement', 'delta_SwitchStatement',
    'delta_NLE'
]
calculate_manhattan_distance = lambda tp, n_sample: [cityblock(row.values, tp) for _, row in n_sample.iterrows()]
calculate_cosine_similarity = lambda tp, n_sample: 1 - cosine_similarity([tp], n_sample)[0]
euclidean_similarity = lambda tp, n_sample: [euclidean(row.values, tp) for _, row in n_sample.iterrows()]
valu_diff_distance = lambda tp, n_sample: [n_sample[i] - tp[i] for i in tp if tp[i] != n_sample[i]]


def distance_func(tp, n_sample, negative_with_parent, tp_with_parent):
    ans_distance = []
    for index, row in n_sample.iterrows():
        n_parent = negative_with_parent.iloc[index]
        ans_distance.append((sum((row - tp).values), abs(cityblock(n_parent, list(tp_with_parent.values())))))
    return ans_distance


def check_valid(value_diff_key, features_affect_key):
    if value_diff_key >= 0 and 0 <= features_affect_key <= value_diff_key:
        return True
    elif value_diff_key <= 0 and 0 >= features_affect_key >= value_diff_key:
        return True
    return False


def apply_rules(value_diff, tp_sample, ignore=False):
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


def move(tp_sample, negative_samples, model):
    can_change = True
    value_diff_end, tp_sample_without_parent = None, None
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
    return value_diff_end


def write(predict, value_diff):
    with open(f"predict_value_diff_{predict}.csv", 'a') as f:
        writer = csv.writer(f)
        for key, value in value_diff.items():
            writer.writerow([key, value])


def get_similarities(tp_sample, negative_samples, name_project, model, path):
    copy_tp = tp_sample.copy()
    move(tp_sample, negative_samples, model)
    return tp_sample if not all(tp_sample == copy_tp) else None


def play_game(X_test, y_true, y_pred, name_project, model=None, path=None):
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
            results = get_similarities(x, negative_classify, name_project, model, path)
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


if __name__ == '__main__':
    def read(predict):
        dict_value = {}
        with open(f"../Algorithms/predict_value_diff_{predict}.csv", 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                value = dict_value.get(row[0], 0)
                dict_value[row[0]] = float(row[1]) + value
        with open(f"value_diff_{predict}.csv", 'a') as f:
            writer = csv.writer(f)
            for key, value in dict_value.items():
                writer.writerow([key, value])
        return dict_value


    def count_rules(path):
        dict_model = {}
        with open(f"../Algorithms/{path}.csv", 'r') as f:
            reader = csv.reader((l.replace('\0', '') for l in f))
            for row in reader:
                if row:
                    value = dict_model.get(row[0], 0)
                    dict_model[row[0]] = value + 1
        return dict_model


    def write_dict_to_csv(path_csv, dict_data):
        with open(path_csv, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for data in dict_data:
                writer.writerow([data, dict_data[data]])


    dict_model = count_rules("count_rules_model_G")
    print({k: v for k, v in sorted(dict_model.items(), key=lambda item: item[1])})

    # # write_dict_to_csv("count_rules_model_RF.csv", dict_model)
    # dict_model_G = count_rules("count_rules_model_G")
    # print({k: v for k, v in sorted(dict_model.items(), key=lambda item: item[1])})

    # write_dict_to_csv("count_rules_model_RF_G.csv", dict_model_G)
    # sub = {key: dict_model_G[key] - dict_model[key] for key in dict_model_G if key in dict_model}
    # print({k: v for k, v in sorted(sub.items(), key=lambda item: item[1])})

    # print(read(0))
    # print(read(1))

    # projects = [
    #     'cayenne', "",'kylin',"", 'jspwiki',"",
    #     'manifoldcf', "",'commons-lang',"", 'tika',"",
    #     'kafka',"",
    #     'zookeeper',"", 'zeppelin',"", 'shiro', "",'logging-log4j2', "",'activemq-artemis', "",'shindig',"",
    #     'directory-studio',"", 'tapestry-5',"", 'openjpa',"", 'knox',"",
    #     'commons-configuration',"",
    #     'xmlgraphics-batik',"",
    #     'mahout',"",
    #     'deltaspike', "",'openwebbeans', "","commons-collections""",
    # ]
    # list_model, list_G = [], []
    # with open(f"../Algorithms/index_tp_model.csv", 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         list_model.append([i for i in row if i != ''])
    # with open(f"../Algorithms/index_tp.csv", 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         list_G.append(row)
    # for i, p in zip(range(len(list_G)), projects):
    #     if i % 2 != 0 or len(list_G[i]) == 0 or len(list_model[i]) == 0:
    #         continue
    #     print(f"----------------------------{p}----------------------------")
    #     same_tp = set(list_G[i]).intersection(list_model[i])
    #     fail_G = len(same_tp.intersection(list_G[i + 1]))
    #     fail_model = len(same_tp.intersection(list_model[i + 1]))
    #     print(fail_G / len(same_tp))
    #     print(fail_model / len(same_tp))
