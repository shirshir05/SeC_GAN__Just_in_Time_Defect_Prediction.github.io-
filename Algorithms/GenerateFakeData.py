import csv
import os
import pickle
import numpy as np
import io
import sys
from sdv.metrics.tabular import BNLogLikelihood, BNLikelihood, SVCDetection, BinaryDecisionTreeClassifier, \
    LogisticDetection, SVCDetection, BinaryDecisionTreeClassifier, BinaryAdaBoostClassifier, \
    BinaryLogisticRegression, BinaryMLPClassifier, MulticlassDecisionTreeClassifier, MulticlassMLPClassifier, \
    LinearRegression, MLPRegressor, GMLogLikelihood, CSTest, KSTest, KSTestExtended, CategoricalCAP, \
    CategoricalZeroCAP, CategoricalGeneralizedCAP, CategoricalNB, CategoricalKNN, CategoricalRF, CategoricalSVM, \
    CategoricalEnsemble, NumericalLR, NumericalMLP, NumericalSVR, NumericalRadiusNearestNeighbor, ContinuousKLDivergence \
    , DiscreteKLDivergence

import torch
from sdv.tabular import CTGAN
import pandas as pd

sys.path.insert(0, '/home/shir0/GAN_VS_RF')
import variable

metrics = [
    LogisticDetection, SVCDetection,  # Detection Metrics
    GMLogLikelihood,  # Likelihood Metrics
    KSTest]  # Statistical Metrics]

metrics_all_data = [BinaryDecisionTreeClassifier, BinaryAdaBoostClassifier, BinaryLogisticRegression,
                    BinaryMLPClassifier,
                    # Binary Classification Metrics
                    LinearRegression, MLPRegressor,  # Regression Metrics
                    MulticlassDecisionTreeClassifier, MulticlassMLPClassifier,  # Multiclass Classification Metrics
                    ContinuousKLDivergence]


def eval_metric_all_data(sample, example, name_file_data):
    scores = []
    for metric in metrics_all_data:
        ans, error1, normalized_score = None, "", None
        ans = metric.compute(example, sample,
                             target='commit insert bug?')
        normalized_score = metric.normalize(ans)
        scores.append({'name': str(metric),
                       'raw_score': ans,
                       'normalized_score': normalized_score,
                       'min_value': metric.min_value,
                       'max_value': metric.max_value,
                       'goal': metric.goal.name,
                       'error': [error1],
                       })
    eval_CTGAN[f"{project}_{name_file_data}"] = scores


def eval_metrics(sample, example, name_file_data):
    scores = []
    for metric in metrics:
        ans, error1, normalized_score = None, "", None
        try:
            ans = metric.compute(example, sample)
            normalized_score = metric.normalize(ans)
        except Exception as e1:
            error1 = str(e1)
        scores.append({'name': str(metric),
                       'raw_score': ans,
                       'normalized_score': normalized_score,
                       'min_value': metric.min_value,
                       'max_value': metric.max_value,
                       'goal': metric.goal.name,
                       'error': [error1],
                       })
    eval_CTGAN[f"{project}_{name_file_data}"] = scores


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def create_data(name_file_ctgan, example, name_file_data):
    try:
        model = CTGAN.load(f"../Data/CTGAN/{name_file_ctgan}")
    except Exception as e:
        print(e)
        with open(f"../Data/CTGAN/{name_file_ctgan}", 'rb') as f:
            model = CPU_Unpickler(f).load()
            model._model._device = 'cpu'
    sample = model.sample(example.shape[0])
    sample = sample.loc[:, ~sample.columns.duplicated()]
    # print(sample.shape)
    if "all" not in name_file_data:
        if "nbug" in name_file_data:
            sample['commit insert bug?'] = 0
        else:
            sample['commit insert bug?'] = 1
    sample.to_csv(os.path.join(NAME_PROJECT, "train_test", f"new_fake_data_{project}_{name_file_data}.csv"))
    eval_metrics(sample, example, name_file_data)
    return sample


if __name__ == '__main__':
    eval_CTGAN = {}
    projects = ['cayenne', 'kylin', 'jspwiki', 'manifoldcf', 'commons-lang', 'tika', 'kafka',
                'zookeeper', 'zeppelin', 'shiro', 'logging-log4j2', 'activemq-artemis', 'shindig',
                'directory-studio', 'tapestry-5', 'openjpa', 'knox', 'commons-configuration', 'xmlgraphics-batik',
                'mahout', 'deltaspike', 'openwebbeans', "commons-collections"]
    projects = ['cayenne', 'knox', 'xmlgraphics-batik', 'mahout', 'deltaspike', 'openwebbeans', 'commons-collections']
    for project in projects:
        try:
            NAME_PROJECT = "../Data/" + project
            df = pd.read_csv(NAME_PROJECT + f"/train_test/train_{project}.csv")
            df = df.iloc[:, 1:]
            X_train = df
            features_check = [col for col in X_train.columns if col in variable.features_check_before_pre_process]
            X_train = X_train[features_check + ['commit insert bug?']]

            # TODO: 2 Generators
            # number_nbug = X_train.iloc[np.where(X_train['commit insert bug?'] == 0)[0]]
            # number_bug = X_train.iloc[np.where(X_train['commit insert bug?'] == 1)[0]]
            #
            # synthetic_nbug = create_data(f'two_G/CTGAN_{project}_nbug_500_512.pkl', number_nbug, "nbug_500_512")
            # synthetic_bug = create_data(f'two_G/CTGAN_{project}_bug_1000_512.pkl', number_bug, "bug_1000_512")
            # all_data = pd.concat([synthetic_nbug, synthetic_bug])

            # TODO: 1 Generator
            all_data = create_data(f'Generator_{project}_1000_512.pkl', X_train, "all_1000_512")

            eval_metric_all_data(all_data, X_train, "all")
        except Exception as e:
            print(e)
            pass
    print(eval_CTGAN)
    with open(os.path.join("..", "Data", "eval_CTGAN_all_1000_512.csv"), "a") as f:
        writer = csv.writer(f)
        writer.writerow(['Name project', 'name', 'raw_score', 'normalized_score', 'min_value', 'max_value', 'goal',
                         'error'])
        for key in eval_CTGAN:
            # f.write(str(key))
            # f.write(",")
            for i in eval_CTGAN[key]:
                f.write(str(key))
                f.write(",")
                for j in i:
                    f.write(str(i[j]))
                    f.write(",")
                f.write("\n")
