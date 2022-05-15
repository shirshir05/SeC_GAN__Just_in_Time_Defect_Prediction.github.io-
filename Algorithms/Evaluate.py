import json
import os
import sys

import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import variable

sys.path.insert(0, '/home/shir0/GAN_VS_RF')
from Play import play_algorithm as Play


# region define threshold


def define_threshold_tesnsor(predictions_proba, y_test):
    # # TODO: v1
    # fpr, tpr, thresholds = roc_curve(y_test, predictions_proba)
    # J = tpr - fpr
    # ix = np.argmax(J)
    # best_thresh = thresholds[ix]
    # predictions = np.where(predictions_proba < best_thresh, 0, 1)
    # print(evaluate_on_test(y_test, predictions, None, predictions_proba, tensor=True))

    # # TODO: v2
    # gmean = np.sqrt(tpr * (1 - fpr))
    # index = np.argmax(gmean)
    # thresholdOpt = round(thresholds[index], ndigits=4)
    # predictions = np.where(predictions_proba < thresholdOpt, 0, 1)
    # print(evaluate_on_test(y_test, predictions, None, predictions_proba, tensor=True))

    # TODO: v3 - 0.5
    predictions = np.where(predictions_proba < 0.5, 0, 1)
    print(evaluate_on_test(y_test, predictions, None, predictions_proba, tensor=True))

    # # TODO v4:
    # precision, recall, thresholds = precision_recall_curve(y_test, predictions_proba)
    # fscore = (2 * precision * recall) / (precision + recall)
    # index = np.argmax(fscore)
    # thresholdOpt = round(thresholds[index], ndigits=4)
    # predictions = np.where(predictions_proba < thresholdOpt, 0, 1)
    # print(evaluate_on_test(y_test, predictions, None, predictions_proba, tensor=True))


def define_threshold(classes, predictions_proba, y_test):
    prob = predictions_proba[1]
    # TODO: v1
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    predictions = np.where(prob < best_thresh, 0, 1)
    print("Max tpr - fpr")
    print(evaluate_on_test(y_test, predictions, classes, predictions_proba))

    # TODO: v2
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    predictions = np.where(prob < thresholdOpt, 0, 1)
    print("tpr * (1 - fpr)")
    print(evaluate_on_test(y_test, predictions, classes, predictions_proba))

    # TODO: v3 - 0.5
    predictions = np.where(prob < np.float64(0.5), 0, 1)
    print("0.5")
    print(evaluate_on_test(y_test, predictions, classes, predictions_proba))

    # TODO v4:
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits=4)
    predictions = np.where(prob < thresholdOpt, 0, 1)
    print("Max fscore")
    print(evaluate_on_test(y_test, predictions, classes, predictions_proba))


# endregion

def plot_old(nj):
    def show_plot(list_values, label, title):
        for i, l in zip(list_values, label):
            plt.plot(i, label=l)
        plt.title(f"{project} {title}")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()
        plt.clf()

    d_loss = pd.read_csv(os.path.join(NAME_PROJECT, "Results", "LOSS", nj + "_real.csv"), header=None)
    show_plot([d_loss[0]], ['real'], "loss D")

    g_loss = pd.read_csv(os.path.join(NAME_PROJECT, "Results", "LOSS", nj + "_fake.csv"), header=None)
    show_plot([d_loss[0], g_loss[0]], ['real', 'fake'], "loss D")


def show_plot(list_values, label, title):
    for i, l in zip(list_values, label):
        plt.plot(i, label=l)
    plt.title(f"{project} {title}")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()
    plt.clf()


def plot(nj, G=False):
    d_loss = pd.read_csv(os.path.join(NAME_PROJECT, "Results", "LOSS", nj + "_D_loss.csv"), header=None,
                         names=[i for i in range(1, 9)])
    average[0].append(d_loss)
    d_bug_real_loss = d_loss[1]
    d_valid_real_loss = [float(i) for i in d_loss[2]]
    d_bug_fake_nb_loss = d_loss[3]
    d_valid_fake_nb_loss = d_loss[4]
    # show_plot([d_bug_real_loss], ["real"], "D loss predict bug data")
    # show_plot([d_valid_real_loss], ["real"], "D loss valid data")
    if G:
        g_nb_loss = pd.read_csv(os.path.join(NAME_PROJECT, "Results", "LOSS", nj + "_G_nb.csv"), header=None,
                                names=[i for i in range(1, 3)])
        g_bug_fake_nb_loss = g_nb_loss[1]
        g_valid_fake_nb_loss = g_nb_loss[2]

        g_b_loss = pd.read_csv(os.path.join(NAME_PROJECT, "Results", "LOSS", nj + "_G_b.csv"), header=None,
                               names=[i for i in range(1, 3)])
        g_bug_fake_b_loss = g_b_loss[1]
        g_valid_fake_b_loss = g_b_loss[2]

        # show_plot([d_bug_fake_nb_loss, d_bug_fake_b_loss], ["not bug", "bug"], "D loss fake data")
        # show_plot([g_bug_fake_nb_loss, g_bug_fake_b_loss], ["not bug", "bug"], "G loss")
        #
        # show_plot([g_valid_fake_nb_loss, d_valid_fake_nb_loss[100:].reset_index()[4]], ['G loss', 'D loss'],
        #           "G not bug fake/real")
        # show_plot([g_valid_fake_b_loss, d_valid_fake_b_loss[100:].reset_index()[6]], ['G loss', 'D loss'],
        #           "G bug fake/real")
        average[1].append(g_nb_loss)
        average[2].append(g_b_loss)


def evaluate_on_test(y_true, y_pred, classes, predicitons_proba, tensor=False):
    def pr_auc_score(y_true, y_score):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
        return metrics.auc(recall, precision)

    scores = {}
    if tensor:
        y_prob_true = predicitons_proba
    else:
        try:
            y_prob_true = dict(zip(classes, predicitons_proba))['1']
        except:
            y_prob_true = dict(zip(classes, predicitons_proba))[1]
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


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    predictions = np.where(y_pred < 0.5, 0, 1)
    df_metrics = evaluate_on_test(y_true, y_pred, None, predictions, True)
    df_metrics = pd.DataFrame.from_dict(df_metrics, orient='index')
    # df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=True)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def evaluate(name, X_test, scaler, y_test, D, NAME_PROJECT, play, project, CTGAN=None,
             threshold=0.5):  # name = after / before
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


def pca_(real, fake):
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    pca.fit(real)
    a = pca.transform(real, )
    b = pca.transform(fake)
    plt.scatter(a[:, 0], a[:, 1], label="real")
    plt.scatter(b[:, 0], b[:, 1], label="fake")
    plt.legend()
    plt.show()


def visualization(train_bug, train_not_bug, G_bug, G_not_bug, project):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    list_G_bug = TSNE(n_components=2).fit_transform(G_bug)
    list_train_bug = TSNE(n_components=2).fit_transform(train_bug)
    list_G_not_bug = TSNE(n_components=2).fit_transform(train_not_bug)
    list_train_not_bug = TSNE(n_components=2).fit_transform(G_not_bug)

    plt.scatter(list_G_bug[:, 0], list_G_bug[:, 1], c='r', label='G bug')
    plt.scatter(list_train_bug[:, 0], list_train_bug[:, 1], c='g', label='real bug')
    plt.title(f"{project} BUG")
    plt.legend()
    plt.savefig("../Data/" + project + "/Results/Visualization/scatter_bug.png")
    plt.show()
    plt.clf()

    plt.scatter(list_G_not_bug[:, 0], list_G_not_bug[:, 1], c='b', label='G not bug')
    plt.scatter(list_train_not_bug[:, 0], list_train_not_bug[:, 1], c='c', label='real not bug')
    plt.title(f"{project} not bug")
    plt.legend()
    plt.savefig("../Data/" + project + "/Results/Visualization/scatter_not_bug.png")
    plt.show()
    plt.clf()


def read_data():
    def read_help(path):
        df = pd.read_csv(NAME_PROJECT + '/train_test/' + path)
        df = df.iloc[:, 1:]
        y_train = df.pop('commit insert bug?')
        return df, y_train

    X_train, y_train = read_help(f"train_{p}.csv")  # TODO: _current
    return X_train, y_train


if __name__ == '__main__':
    projects = ['cayenne', 'kylin', 'jspwiki', 'manifoldcf', 'commons-lang', 'tika', 'kafka',
                'zookeeper', 'zeppelin', 'shiro', 'logging-log4j2', 'activemq-artemis', 'shindig',
                'directory-studio', 'tapestry-5', 'openjpa', 'knox', 'commons-configuration', 'xmlgraphics-batik',
                'mahout', 'deltaspike', 'openwebbeans', "commons-collections"]
    # TODO: visualization
    # for project in projects:
    #     try:
    #         NAME_PROJECT = f"../Data/{project}"
    #         p = project
    #         X_train, y_train = read_data()
    #         X = np.expand_dims(X_train, axis=-1)
    #         Y = np.expand_dims(y_train, axis=1)
    #         x_zero_only = X[np.where(Y == [0])[0]]
    #         y_zero_only = Y[np.where(Y == [0])[0]]
    #         x_one_only = X[np.where(Y == [1])[0]]
    #         y_one_only = Y[np.where(Y == [1])[0]]
    #
    #         data = pd.read_csv(os.path.join(NAME_PROJECT, "train_test", f"new_fake_data_{p}_bug_1000_512.csv"))
    #         data = data.iloc[:, 1:]
    #         data.pop('commit insert bug?')
    #         data_nbug = pd.read_csv(os.path.join(NAME_PROJECT, "train_test", f"new_fake_data_{p}_nbug_500_512.csv"))
    #         data_nbug = data_nbug.iloc[:, 1:]
    #         data_nbug.pop('commit insert bug?')
    #
    #         visualization(x_one_only[:,:,0], x_zero_only[:,:,0], data, data_nbug, project)
    #     except Exception as e:
    #         print(e)

    # average = [[], [], [], [], [], []]
    # for project in projects:
    #     try:
    #         # variable.projects
    #         NAME_PROJECT = "../Data/" + project
    #         # plot('94')
    #         plot_old("133")
    #     except Exception as e:
    #         print(e)
    #         pass
    # project = "all"
    # s = pd.concat([i for i in average[0]], axis=1)
    # s = s.groupby(s.columns.values, axis=1).mean()
    # show_plot([s[1], s[3]], ['real', 'fake'], "loss D predict") # TODO: need change index after 118
    # show_plot([s[2], s[4]], ['real', 'fake'], "loss D validation")
    #
    # g_loss_bug = pd.concat([i for i in average[2]], axis=1)
    # g_loss_bug = g_loss_bug.groupby(g_loss_bug.columns.values, axis=1).mean()
    #
    # g_loss_nb = pd.concat([i for i in average[1]], axis=1)
    # g_loss_nb = g_loss_nb.groupby(g_loss_nb.columns.values, axis=1).mean()
    #
    # show_plot([g_loss_nb[1], g_loss_bug[1],  s[4]], ["not bug G", "bug G", "real"], "validation")
    # show_plot([g_loss_nb[2], g_loss_bug[2]], ["not bug G", "bug G"], "G loss predict")
