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
    'resources/machine_learning_results/models/BalancedDecisionTree'
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
plt.show(block=False)


target = 'Propensity'
test = (
    pd.read_csv('resources/generated_data/nest_features_test.csv')
    [list(model.feature_names_in_) + [target]]
)
test_x, test_y = extract_target_feature(test, target)
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
