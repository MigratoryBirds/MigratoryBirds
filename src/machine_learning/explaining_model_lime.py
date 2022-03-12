import sys
sys.path.append('src')
import lime.lime_tabular
import sklearn
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from machine_learning.utils import extract_target_feature

columns_to_drop = [
    'ShyBirdsPercentage_Nearby_200',
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

model: sklearn.tree.DecisionTreeClassifier = joblib.load(
    'resources/machine_learning_results/models/BalancedDecisionTree'
)

train = (
    pd.read_csv('resources/generated_data/nest_features_train.csv')
    .drop(columns=columns_to_drop)
)
test = (
    pd.read_csv('resources/generated_data/nest_features_test.csv').
    drop(columns=columns_to_drop)
)

train_x, train_y = extract_target_feature(train, 'Propensity')
test_x, test_y = extract_target_feature(test, 'Propensity')

explainer = lime.lime_tabular.LimeTabularExplainer(
    train_x.values,
    feature_names=model.feature_names_in_,
    class_names=model.classes_,
    discretize_continuous=True
)

explained_idx = 9

print(dict(test_x.iloc[explained_idx]))
print(f'label: {test_y[explained_idx]}')

exp = explainer.explain_instance(
    test_x.iloc[explained_idx],
    model.predict_proba,
    num_features=model.n_features_in_
)

ax = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
