import csv
from copy import copy
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from variable import features_check_before_pre_process


class Preprocessing:

    def __init__(self, NAME_PROJECT, name_file, p):
        self.df = pd.read_csv(os.path.join(NAME_PROJECT, name_file))
        self.path = os.path.join(NAME_PROJECT, name_file)
        self.Y = None
        self.X = None
        self.name_project = NAME_PROJECT
        self.project = p

    def main(self):
        self.preprocessing()
        self.Y = self.df.pop('commit insert bug?')
        self.X = self.df
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_val, self.y_val = self.split_train_test()
        self.write_data()

    def preprocessing(self):
        def save_info():
            with open(f"statistical/statistical_{self.project}.csv", 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(
                    [self.df.shape[0], self.df.shape[1], self.df['commit insert bug?'].sum() / self.df.shape[0]])

        # Remove test file
        def filter_fn(row):
            if "test" in row['file_name'].lower():
                return False
            else:
                return True

        self.df.rename(columns={'blame commit': 'commit insert bug?'}, inplace=True)
        save_info()  # No filter
        # ignore large commits those that change at least 100 files
        count_file = self.df.groupby(
            ['commit']).count()  # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7898457
        remove_commit = [index for index, row in count_file.iterrows() if row['file_name'] >= 100]
        self.df = self.df[~self.df['commit'].isin(remove_commit)]
        save_info()  # 1- Too many files
        # ignore large commits those that change at least 10,000 lines
        self.df = self.df[self.df['current_used_lines'] < 10000]
        save_info()  # 2- Too much churn

        # Remove redundant columns
        self.df = self.df.iloc[:, 1:]  # commit
        del self.df['added_lines+removed_lines']
        del self.df['added_lines-removed_lines']

        # Remove commit that doesn't change lines
        self.df = self.df[self.df['used_added_lines+used_removed_lines'] != 0]
        # Remove commit that doesn't change lines
        self.df = self.df[self.df['current_methods_count'] != 0]
        save_info()  # 3- Remove modifications without functional change

        self.df = self.df.loc[(self.df['mode'] == 'M') | (self.df['mode'] == 'A')]
        save_info()  # 4- Remove D or R files

        self.df = self.df.loc[(self.df['is_java'] == True)]
        # print("Step 6 - Select java file")

        features_to_drop = ['adhoc', 'MATH-']  # parent # 'current',
        self.df = self.df.drop(
            columns=list(filter(lambda c: any(map(lambda f: f in c, features_to_drop)), self.df.columns)), axis=1)
        self.df = self.df.loc[(self.df['is_test'] == False)]

        self.df = self.df[self.df.apply(filter_fn, axis=1)]
        del self.df["file_name"]
        save_info()  # 5 - test

        self.df = self.df[features_check_before_pre_process + ['commit insert bug?']]
        # Remove select columns

        self.df.dropna(inplace=True)  # nan return from pattern features
        save_info()  # 6 - dropna
        self.df = self.df.drop_duplicates()
        save_info()  # 7 - duplicate

    def split_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=12,
                                                            stratify=self.Y)
        X_train.replace(0, np.nan, inplace=True)
        X_train.dropna(axis=1, how='any', thresh=0.05 * X_train.shape[1], inplace=True)
        X_train.replace(np.nan, 0, inplace=True)
        print("Step 14 - Remove columns 95% zero from train")
        print(X_train.shape)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=12,
                                                          stratify=y_train)
        col_drop = [col for col in X_test.columns if col not in X_train]
        X_test = X_test.drop(col_drop, axis=1)

        return X_train, X_test, y_train, y_test, X_val, y_val

    def write_data(self):
        self.X_train['commit insert bug?'] = self.y_train
        self.X_train.to_csv(self.name_project + f"/train_test/train_{self.project}.csv")

        self.X_test['commit insert bug?'] = self.y_test
        self.X_test.to_csv(self.name_project + f"/train_test/test_{self.project}.csv")

        self.X_val['commit insert bug?'] = self.y_val
        self.X_val.to_csv(self.name_project + f"/train_test/val_{self.project}.csv")
