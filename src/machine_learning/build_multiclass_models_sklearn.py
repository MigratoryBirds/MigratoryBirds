"""
This class inherits from the build_models_sklearn_template to build
multiclass models on the dataset using both the original dataset as well
as the features after going through PCA

When evaluating, it is also evaluated based on binary classification
"""


from typing import Any
from utils import \
    scale_features, print_stdout_and_file, plot_confusion_matrix, perplexity
from build_models_sklearn_template import \
    BuildModelsSklearnTemplate
from sklearn.metrics import \
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from models import classification_models as models_dict
np.random.seed(42)


class BuildMulticlassModelsSklearn(BuildModelsSklearnTemplate):
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
        self.df_train.drop(columns=self.columns_to_drop, inplace=True)
        self.df_test.drop(columns=self.columns_to_drop, inplace=True)

    def _do_preprocessing(self) -> None:
        self.x_train, self.x_test = scale_features(
            self.x_train, self.x_test, self.columns_to_scale
        )
        BuildModelsSklearnTemplate._do_preprocessing(self)
        self.x_train, self.x_test = self.x_train, self.x_test
        if self.pca:
            pca = PCA(n_components=19, random_state=1)
            self.x_train = pca.fit_transform(self.x_train)
            self.x_test = pca.transform(self.x_test)

    def _do_print_evaluations(
        self,
        model_name: str,
        classifier: sklearn.base.BaseEstimator,
    ) -> None:
        train_predictions = classifier.predict(self.x_train)
        test_predictions = classifier.predict(self.x_test)
        positive_label_index \
            = list(classifier.classes_).index(self.positive_label)
        train_predictions_proba = (
            1
            - classifier.predict_proba(self.x_train)
                [:, positive_label_index]
        )
        test_predictions_proba = (
            1
            - classifier.predict_proba(self.x_test)
                [:, positive_label_index]
        )
        print_stdout_and_file(
            '\t\t\taccuracy on train: '
            f'{accuracy_score(self.y_train.values, train_predictions)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmacro f1 on train: '
            f'{f1_score(self.y_train.values, train_predictions, average="macro")}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmicro f1 on train: '
            f'{f1_score(self.y_train.values, train_predictions, average="micro")}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\taccuracy on test: '
            f'{accuracy_score(self.y_test.values, test_predictions)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmacro precision on test: '
            f'{precision_score(self.y_test.values, test_predictions, average="macro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmicro precision on test: '
            f'{precision_score(self.y_test.values, test_predictions, average="micro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmacro recall on test: '
            f'{recall_score(self.y_test.values, test_predictions, average="macro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmicro recall on test: '
            f'{recall_score(self.y_test.values, test_predictions, average="micro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmacro f1 on test: '
            f'{f1_score(self.y_test.values, test_predictions, average="macro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmicro f1 on test: '
            f'{f1_score(self.y_test.values, test_predictions, average="micro", zero_division=0)}',
            self.file_pointer
        )
        plot_confusion_matrix(
            self.y_test.values,
            test_predictions,
            self.df_train[self.target_column].unique(),
            f"multiclass_{model_name}_average_confusion_matrix.png",
        )
        print_stdout_and_file("", self.file_pointer)
        binary_y_train_actual = self.y_train.values
        binary_y_train_predicted = train_predictions
        binary_y_test_actual = self.y_test.values
        binary_y_test_predicted = test_predictions
        print_stdout_and_file(
            '\t\t\tbinary accuracy on train: '
            f'{accuracy_score(binary_y_train_actual, binary_y_train_predicted)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary f1 on train: '
            f'{f1_score(binary_y_train_actual, binary_y_train_predicted, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary perplexity train: '
            f'{perplexity(train_predictions_proba, binary_y_train_predicted, self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary accuracy on test: '
            f'{accuracy_score(binary_y_test_actual, binary_y_test_predicted)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary precision on test: '
            f'{precision_score(binary_y_test_actual, binary_y_test_predicted, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary precision on test: '
            f'{precision_score(binary_y_test_actual, binary_y_test_predicted, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary recall on test: '
            f'{recall_score(binary_y_test_actual, binary_y_test_predicted, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary f1 on test: '
            f'{f1_score(binary_y_test_actual, binary_y_test_predicted, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary perplexity test: '
            f'{perplexity(test_predictions_proba, binary_y_test_predicted, self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary roc_auc_score test: '
            f'{roc_auc_score(binary_y_test_actual, binary_y_test_predicted)}',
            self.file_pointer
        )
        plot_confusion_matrix(
            self.y_test.values,
            test_predictions,
            self.df_train[self.target_column].unique(),
            f"multiclass_binary_{model_name}_average_confusion_matrix.png",
        )


process = BuildMulticlassModelsSklearn(
    models_dict,
    input_train_csv_file_name
        ='resources/generated_data/nest_features_train.csv',
    input_test_csv_file_name
        ='resources/generated_data/nest_features_test.csv',
    target_column='Propensity',
    output_file_name=(
        'resources/machine_learning_results/'
        'multiclass_classification_models.txt'
    ),
    columns_to_drop=[
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
