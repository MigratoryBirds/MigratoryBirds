"""
Json formatted details for hyper parameter tuning
sklearn models
"""

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


pow_10_paramter = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

classification_models = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "hyperparameters": {
            "C": pow_10_paramter,
        },
        "set_parameters": {
            "max_iter": 9999,
            "solver": "liblinear"
        }
    },
    "BalancedLogisticRegression": {
        "class": LogisticRegression,
        "hyperparameters": {
            "C": pow_10_paramter,
        },
        "set_parameters": {
            "max_iter": 9999,
            "class_weight": "balanced",
            "solver": "liblinear"
        }
    },
    "NaiveBayes": {
        "class": GaussianNB,
        "hyperparameters": {},
        "set_parameters": {}
    },
    "MLPClassifier": {
        "class": MLPClassifier,
        "hyperparameters": {
            "hidden_layer_sizes": [
                (500, 500), (300, 200), (200, 200, 200), (100, 100)
            ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "alpha": pow_10_paramter,
        },
        "set_parameters": {
            "early_stopping": True,
            "max_iter": 500,
        }
    },
    "SVM": {
        "class": SVC,
        "hyperparameters": {
            "C": pow_10_paramter,
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": list(range(1, 6, 1)),
        },
        "set_parameters": {
            "probability": True
        }
    },
    "BalancedSVM": {
        "class": SVC,
        "hyperparameters": {
            "C": pow_10_paramter,
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": list(range(1, 3, 1)),
        },
        "set_parameters": {
            "class_weight": "balanced",
            "probability": True
        }
    },
    "DecisionTree": {
        "class": DecisionTreeClassifier,
        "hyperparameters": {
            "max_depth": [None, 30, 60, 100, 200, 300, 500, 1000],
            "max_leaf_nodes": [None, 30, 60, 100, 200, 300, 500, 1000]
        },
        "set_parameters": {}
    },
    "BalancedDecisionTree": {
        "class": DecisionTreeClassifier,
        "hyperparameters": {
            "max_depth": [None, 30, 60, 100, 200, 300, 500, 1000],
            "max_leaf_nodes": [None, 30, 60, 100, 200, 300, 500, 1000]
        },
        "set_parameters": {
            "class_weight": "balanced"
        }
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "hyperparameters": {
            "max_depth": [None, 30, 60, 100, 200, 300, 500, 1000],
            "max_leaf_nodes": [None, 30, 60, 100, 200, 300, 500, 1000]
        },
        "set_parameters": {}
    },
    "BalancedRandomForest": {
        "class": RandomForestClassifier,
        "hyperparameters": {
            "max_depth": [None, 30, 60, 100, 200, 300, 500, 1000],
            "max_leaf_nodes": [None, 30, 60, 100, 200, 300, 500, 1000]
        },
        "set_parameters": {
            "class_weight": "balanced"
        }
    }
}
