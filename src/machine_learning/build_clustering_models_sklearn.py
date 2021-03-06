"""
This class inherits BuildModelsSklearnTemplate to build unsupervised
models to split the data into 2 in order to try to find patterns and
different splits in the dataset which supervised learning may not
have been able to capture

This class tries to find a better propensity value than the one
provided by the researchers, which was 20
"""


from typing import Any
from build_models_sklearn_template import \
    BuildModelsSklearnTemplate
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from models import clustering_models as models_dict
from machine_learning.utils import (
    scale_features,
    extract_target_feature,
    print_stdout_and_file,
)
from sklearn.tree import DecisionTreeClassifier
np.random.seed(42)


class BuildClusteringModelsSklearn(BuildModelsSklearnTemplate):
    def __init__(
        self,
        models: dict,
        input_train_csv_file_name: str,
        input_test_csv_file_name: str,
        target_column: str,
        output_file_name: str,
        columns_to_drop: list[str] = [],
        train_folds: int = 5,
        tuning_iterations: int = 20,
        pca=False,
        positive_label: Any = 1,
        columns_to_scale: list[str] = [],
    ):
        BuildModelsSklearnTemplate.__init__(
            self,
            models,
            input_train_csv_file_name,
            input_test_csv_file_name,
            target_column,
            output_file_name,
            train_folds=train_folds,
            tuning_iterations=tuning_iterations,
            positive_label=positive_label,
        )
        self.columns_to_drop = columns_to_drop
        self.pca = pca
        self.columns_to_scale = columns_to_scale

    def _do_at_init(self) -> None:
        self.df = pd.concat([self.df_train, self.df_test])
        self.df.drop(columns=self.columns_to_drop, inplace=True)

    def _initialize_train_test_split(self) -> None:
        self.x, self.y \
            = extract_target_feature(self.df, self.target_column)

    def _do_preprocessing(self) -> None:
        self.x, _ = scale_features(
            self.x, None, self.columns_to_scale
        )
        if self.pca:
            pca = PCA(n_components=19, random_state=1)
            self.x = pca.fit_transform(self.x)

    def compute(self) -> None:
        self._do_at_init()
        self._initialize_train_test_split()
        self._do_preprocessing()
        for name, model in self.models.items():
            print_stdout_and_file(f'Now training {name}', self.file_pointer)
            print('\tFitting...')
            classifier = model['class'](**model['set_parameters'])
            labels = classifier.fit_predict(self.x)
            self.df['clustering_label'] = labels
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(
                self.df['SnapsRasps'].values.reshape(-1, 1),
                self.df['clustering_label']
            )
            print(f'\tbest split: {tree.tree_.threshold[0]}')
        self.file_pointer.close()

    def _do_print_evaluations(
        self,
        model_name: str,
        classifier: sklearn.base.BaseEstimator,
    ) -> None:
        train_predictions = classifier.predict(self.x_train)
        test_predictions = classifier.predict(self.x_test)
        print_stdout_and_file(
            '\t\t\tr^2 on train: '
            f'{r2_score(self.y_train.values, train_predictions)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tr^2 on test: '
            f'{r2_score(self.y_test.values, test_predictions)}',
            self.file_pointer
        )


process = BuildClusteringModelsSklearn(
    models=models_dict,
    input_train_csv_file_name
        ='resources/generated_data/nest_features_train.csv',
    input_test_csv_file_name
        ='resources/generated_data/nest_features_test.csv',
    target_column='SnapsRasps',
    output_file_name=(
        'resources/machine_learning_results/'
        'regression_models.txt'
    ),
    columns_to_drop=[
        'NestID',
        'Year',
        'Laydate_first_egg',
        'Date_trial',
        'Days_from_LD',
        'Rasps',
        'Bill_snaps',
        # 'SnapsRasps',
        'Propensity',
        'Propensity_0',
        'Propensity_17.5',
        'Propensity_19.5',
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
    ],
    columns_to_scale=[
        'Cuckoo_perch',
        'x',
        'y',
        'z',
        'nests_nearby_15',
        'nests_nearby_30',
        'nests_nearby_50',
        'nests_nearby_100',
        'nests_nearby_200',
        'nests_nearby_300',
        'closest_nest_distance',
        'ClusterSize_15',
        'ClusterSize_30',
        'ClusterSize_50',
        'ClusterSize_100',
        'ClusterSize_200',
        'ClusterSize_300',
        'ShyBirdsPercentage_Clusters_15',
        'ShyBirdsPercentage_Clusters_30',
        'ShyBirdsPercentage_Clusters_50',
        'ShyBirdsPercentage_Clusters_100',
        'ShyBirdsPercentage_Clusters_200',
        'ShyBirdsPercentage_Clusters_300',
        'ShyBirdsPercentage_Nearby_15',
        'ShyBirdsPercentage_Nearby_30',
        'ShyBirdsPercentage_Nearby_50',
        'ShyBirdsPercentage_Nearby_100',
        'ShyBirdsPercentage_Nearby_200',
        'ShyBirdsPercentage_Nearby_300',
    ],
    train_folds=5,
    tuning_iterations=20,
    positive_label=1,
)
process.compute()
