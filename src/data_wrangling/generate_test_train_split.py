"""
This script divides the original dataset into its own train and test set
The split is 80% train set and 20% for the test set
"""

import sys
sys.path.append('src')
import numpy as np
import pandas as pd
from machine_learning.utils import get_stratified_train_test_folds
np.random.seed(42)


df = pd.read_csv('resources/generated_data/nest_features.csv')
train_df, test_df = \
    get_stratified_train_test_folds(df, 'Propensity', test_factor=0.2)
train_df.to_csv(
    'resources/generated_data/nest_features_train.csv', index=False
)
test_df.to_csv('resources/generated_data/nest_features_test.csv', index=False)
