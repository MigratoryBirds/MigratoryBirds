import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import sklearn

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
