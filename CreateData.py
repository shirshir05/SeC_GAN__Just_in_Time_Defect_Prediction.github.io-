import os
import pandas as pd
import variable
from Preprocess.preprocessing import Preprocessing
from SZZ.issues_extractor import main_szz


def read_commit_blame():
    """
    Reading the file that contains the instances (commit + file) that induced defect.
    This file Written in call the SZZ algorithm.

    :return: dataframe contain two columns: commit and file name
    :rtype: DataFrame

    .. note::
            You must run the function main_szz() in file issues_extractor.py before execute this function
    """
    dataset = pd.read_csv(
        variable.get_name_dit_blame() + r"/pydriller_" + variable.get_name_github() + "_bugfixes_bic.csv",
        delimiter=',')
    dataset['filename'] = dataset['filename'].str.replace("\\", r'/')
    del dataset['bugfix_commit']
    return dataset.drop_duplicates()


def update_bug(df, commit_blame):
    """
    The function merges between the extracted features (df) to the column whether the change induced a bug
    (commit_blame).

    :param df: all data that extract from bic repository
    :type df: DataFrame
    :param commit_blame: dataframe contain two columns: commit and file name. all commit in this file induced defect
    according SZZ algorithm.
    :type commit_blame: DataFrame
    :return: dataset add column name "blame commit" to df that indicate if the modification (commit+file) induced defect.
    :rtype: DataFrame

    .. notes:: You must run the function  main_szz() in file issues_extractor.py before execute this function

    """
    dataset = df.merge(commit_blame, how='left', left_on=['file_name', 'commit'], right_on=['filename', 'bic'])
    dataset['blame commit'] = dataset.apply(lambda x: 1 if len(str(x['bic'])) > 3 else 0, axis=1)
    del dataset["bic"]
    dataset.to_csv(NAME_PROJECT + "/all_features.csv", index=None)
    return dataset


def add_modification(dataset):
    """
    The file "modification_commit.csv" written in SZZ algorithm process.
    The file contains the change (rename (R), added (A), deleted(D) or modify(M)) that made each modification.
    The function merges between the dataset and the file  "modification_commit.csv"
    The function writes the dataset to a file named "all_features.csv".

    :param df: all data that extract from bic repository
    :type df: DataFrame

    :return: df Dataframe with a new column called "mode" that expresses the change (rename (R), added (A), deleted(D) or
     modify(M))

    .. note:: You must run the function  main_szz() in file issues_extractor.py before execute this function
    """
    modification = pd.read_csv(variable.get_name_dit_blame() + '/modification_commit.csv', delimiter=',')
    # modification['file_name'] = modification['file_name'].str.lower()
    dataset = dataset.merge(modification, how='inner', left_on=['file_name', 'commit'],
                            right_on=['file_name', 'commit_sha'])
    dataset.drop(['commit_sha'], axis=1, inplace=True)
    dataset.to_csv(NAME_PROJECT + "/all_features.csv", index=None)
    return dataset


def create_directories():
    """
    The function that produces the work environment for the project.
    The function performs a clone from Github to a folder called Repo.
    And creates additional folders to store information as a folder for saving results.
    """
    import git
    if not os.path.exists(os.path.join("..", "Repo", variable.get_name_github())):
        git.Git(os.path.join("..", "Repo")).clone("https://github.com/apache/" + variable.get_name_github() + ".git")
    if not os.path.exists(os.path.join(NAME_PROJECT)):
        os.mkdir(os.path.join(NAME_PROJECT))
    if not os.path.exists(os.path.join(NAME_PROJECT, "train_test")):
        os.mkdir(os.path.join(NAME_PROJECT, "train_test"))
    if not os.path.exists(os.path.join(NAME_PROJECT, "blame")):
        os.mkdir(os.path.join(NAME_PROJECT, "blame"))
    if not os.path.exists(os.path.join(NAME_PROJECT, "Results")):
        os.mkdir(os.path.join(NAME_PROJECT, "Results"))
    if not os.path.exists(os.path.join(NAME_PROJECT, "BEGAN")):
        os.mkdir(os.path.join(NAME_PROJECT, "BEGAN"))
    if not os.path.exists(os.path.join(NAME_PROJECT, "load_model")):
        os.mkdir(os.path.join(NAME_PROJECT, "load_model"))
    if not os.path.exists(os.path.join("Tuning_Results")):
        os.mkdir(os.path.join("Tuning_Results"))


if __name__ == '__main__':
    # TODO: If you want to add a new project you need add the key_issue to the file named variable.py

    projects = ['cayenne', 'kylin', 'jspwiki', 'manifoldcf', 'commons-lang', 'tika', 'kafka',
                'zookeeper', 'zeppelin', 'shiro', 'logging-log4j2',
                'activemq-artemis', 'shindig',
                'directory-studio', 'tapestry-5', 'openjpa', 'knox',
                'commons-configuration', 'xmlgraphics-batik',
                'mahout', 'deltaspike', 'openwebbeans', "commons-collections"]
    for p in projects:
        project = p
        with open(os.path.join("name_project.txt"), "w") as f:
            f.write(project)
        NAME_PROJECT = "Data/" + project
        create_directories()
        print(project)
        main_szz(os.path.join("..", "Repo", variable.get_name_github()), variable.get_name_github(),
                 variable.get_key_issue(),
                 variable.get_repo_full_name(), variable.get_name_dit_blame())
        df = pd.read_csv(NAME_PROJECT + "/all.csv")
        commit_blame = read_commit_blame()
        dataset = update_bug(df, commit_blame)
        dataset = add_modification(dataset)
        pre = Preprocessing(NAME_PROJECT, "all_features.csv", p)
        pre.main()
