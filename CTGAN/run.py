import sys
from sdv.tabular import CTGAN
import pandas as pd
import numpy as np
import os
import torch
import variable
torch.manual_seed(420)
np.random.seed(420)


def parameters():
    """
    A function that defines the parameters in the CTGAN model.

    :return: dict_var: key -  name of parameter, value of each parameter
    :rtype: Dict
    """
    dict_var = {'max_value': None, 'min_value': None, 'rounding': 2, 'epochs': 500, 'batch_size': 30,
                'log_frequency': False, 'embedding_dim': 128, 'generator_dim': (256, 256, 256),
                'discriminator_dim': (256, 256, 256), 'discriminator_lr': 2e-4, 'generator_lr': 2e-4,
                'discriminator_decay': 1e-6, 'generator_decay': 1e-6, 'discriminator_steps': 1, 'verbose': False,
                'cuda': True}
    return dict_var


if __name__ == '__main__':
    project = sys.argv[1]
    NAME_PROJECT = f"../Data/{project}/train_test/train_" + project + ".csv"
    print(NAME_PROJECT)

    df = pd.read_csv(NAME_PROJECT)
    df = df.iloc[:, 1:]
    df = df.iloc[:-1]
    features_check = [col for col in df.columns if col in variable.features_check_before_pre_process] + [
        'commit insert bug?']
    X = df[features_check]
    dict_var = parameters()

    # TODO: two generator
    y = X.pop('commit insert bug?')
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(y, axis=1)
    x_zero_only = pd.DataFrame(X[np.where(Y == [0])[0]][:, :, 0], columns=features_check[: -1])
    x_one_only = pd.DataFrame(X[np.where(Y == [1])[0]][:, :, 0], columns=features_check[: -1])

    # model bug
    dict_var['epochs'] = 2000
    model = CTGAN(**dict_var)
    model._metadata.fit(pd.DataFrame(X[:, :, 0], columns=features_check[: -1]))
    model._metadata_fitted = True
    model.fit(x_one_only)
    model.save(os.path.join("model", f'CTGAN_{project}_bug_{dict_var["epochs"]}_.pkl'))

    # model without bug
    dict_var['epochs'] = 500
    model = CTGAN(**dict_var)
    model._metadata.fit(pd.DataFrame(X[:, :, 0], columns=features_check[: -1]))
    model._metadata_fitted = True
    model.fit(x_zero_only)
    model.save(os.path.join("model", f'CTGAN_{project}_nbug_{dict_var["epochs"]}_.pkl'))
