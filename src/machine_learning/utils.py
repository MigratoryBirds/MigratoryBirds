"""
Useful functions to clean up the code for building models
"""

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix


def split_in_folds_regression(df: pd.DataFrame, n, target_feature):
    df_sorted = df.sort_values(target_feature)
    full_index_set = set(range(len(df)))
    remaining_index_set = full_index_set.copy()
    random_chance = n
    folds = []
    for _ in range(n):
        remaining_index_list = np.array(list(remaining_index_set))
        randoms = (
            (np.random.uniform(0, random_chance, len(remaining_index_set)) + 1)
            // random_chance
        )
        indexes = remaining_index_list[randoms == 1]
        remaining_index_set.difference_update(set(indexes))
        folds.append(df_sorted.iloc[indexes])
        random_chance -= 1
    return folds


def split_in_folds_classification(df: pd.DataFrame, n, target_feature):
    skf = StratifiedKFold(n, shuffle=True)
    folds = []
    for _, test_indices in skf.split(df, df[target_feature]):
        folds.append(df.iloc[test_indices])
    return folds


def get_stratified_train_test_folds(
    df: pd.DataFrame, target_feature, test_factor: int = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = train_test_split(df, test_size=test_factor, stratify=df[target_feature])
    train_df, test_df = split[0], split[1]
    return train_df, test_df


def extract_target_feature(df, target_feature):
    target = df[target_feature]
    df = df.drop(columns=[target_feature])
    return df, target


def scale_features(df_train, df_test, features):
    sc = StandardScaler()
    df_train = df_train.copy()
    df_train[features] = sc.fit_transform(df_train[features])
    if df_test is not None:
        df_test = df_test.copy()
        df_test[features] = sc.transform(df_test[features])
    return df_train, df_test


def print_stdout_and_file(string, file_pointer):
    print(string, flush=True)
    print(string, file=file_pointer, flush=True)


def plot_confusion_matrix(actual, predicted, labels, fig_name):
    cm = confusion_matrix(actual, predicted, labels=labels)
    cmap = LinearSegmentedColormap.from_list("", ["white", "darkBlue"])
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        vmin=0,
        fmt="d"
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.rcParams["figure.figsize"] = [15, 9]
    plt.savefig(f'resources/machine_learning_results/{fig_name}')
    plt.clf()


def perplexity(y_pred, y_test, positive_label):
    tmp = [
        y_pred[i] if y_test[i] == positive_label
        else 1 - y_pred[i]
        for i in range(len(y_pred))
    ]
    return np.exp(-np.mean(np.log(tmp)))
