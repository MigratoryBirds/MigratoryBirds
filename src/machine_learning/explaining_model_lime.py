import sys
sys.path.append('src')
import lime.lime_tabular
import sklearn
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from machine_learning.utils import extract_target_feature

model: sklearn.tree.DecisionTreeClassifier = joblib.load(
    'resources/machine_learning_results/models/BalancedDecisionTree'
)

target = 'Propensity'
train = (
    pd.read_csv('resources/generated_data/nest_features_train.csv')
    [list(model.feature_names_in_) + [target]]
)
test = (
    pd.read_csv('resources/generated_data/nest_features_test.csv')
    [list(model.feature_names_in_) + [target]]
)

train_x, train_y = extract_target_feature(train, target)
test_x, test_y = extract_target_feature(test, target)

explainer = lime.lime_tabular.LimeTabularExplainer(
    train_x.values,
    feature_names=model.feature_names_in_,
    class_names=model.classes_,
    discretize_continuous=True
)

explained_idx = 49

print(dict(test_x.iloc[explained_idx]))
print(f'label: {test_y[explained_idx]}')
print(f'Predicted: {model.predict([test_x.iloc[explained_idx]])}')

exp = explainer.explain_instance(
    test_x.iloc[explained_idx],
    model.predict_proba,
    num_features=model.n_features_in_
)

ax = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
