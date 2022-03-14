import sys
sys.path.append('src')
import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import sklearn
from sklearn.inspection import permutation_importance
from machine_learning.utils import extract_target_feature

model: sklearn.tree.DecisionTreeClassifier = joblib.load(
    'resources/machine_learning_results/models/DecisionTree'
)

forest_importances = pd.Series(
    model.feature_importances_,
    index=model.feature_names_in_
)
std = np.std(model.feature_importances_)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using Mean Decrease Impurity (MDI)")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()


columns_to_drop = [
    'NestID',
    'Year',
    'Laydate_first_egg',
    'Date_trial',
    'Days_from_LD',
    'Rasps',
    'Bill_snaps',
    'SnapsRasps',
    'Site',
    'lat',
    'long',
    'Date_nest_found',
    'New_rebuild',
    'Rebuild_original',
    'Shape',
    'Nearby',
    'Water_area',
    'ClusterID_15',
    'ClusterID_30',
    'ClusterID_50',
    'ClusterID_100',
    'ClusterID_200',
    'ClusterID_300',
    # 'Propensity',
    'Propensity_0',
    'Propensity_17.5',
    'Propensity_19.5',
]

test = (
    pd.read_csv('resources/generated_data/nest_features_test.csv').
    drop(columns=columns_to_drop)
)

test_x, test_y = extract_target_feature(test, 'Propensity')


result = permutation_importance(
    model, test_x, test_y, n_repeats=10, random_state=42, n_jobs=2
)

forest_importances = pd.Series(
    result.importances_mean, index=model.feature_names_in_
)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
