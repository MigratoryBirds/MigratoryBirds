"""
This script divides the original dataset into its own train and test set
The split is 80% train set and 20% for the test set
"""

import sys
sys.path.append('src')
import numpy as np
import pandas as pd
from machine_learning.utils import \
    get_stratified_train_test_folds, one_hot_encode_and_bind
np.random.seed(42)

features_to_one_hot_encode = [
    'Model',
    # 'Year',
    # 'Shape',
    # 'Nearby',
    # 'Water_area',
    # 'ClusterID_15',
    # 'ClusterID_30',
    # 'ClusterID_50',
    # 'ClusterID_100',
    # 'ClusterID_200',
    # 'ClusterID_300',
]


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> None:
    for col in features_to_one_hot_encode:
        df = one_hot_encode_and_bind(df, col)
    return df


df = pd.read_csv('resources/generated_data/nest_features.csv')
df = one_hot_encode(df, features_to_one_hot_encode)
train_df, test_df = \
    get_stratified_train_test_folds(df, 'Propensity_19.5', test_factor=0.3)
train_df.to_csv(
    'resources/generated_data/nest_features_train.csv', index=False
)
test_df.to_csv('resources/generated_data/nest_features_test.csv', index=False)
