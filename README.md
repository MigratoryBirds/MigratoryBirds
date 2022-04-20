# MigratoryBirds
Project for DSP2022 - Data Science Project I - Detecting micro and macro spatial-temporal patterns in behaviour and habitat selection by a migratory bird

## Python version
* Python version python3.9 or higher

## Libraries
* numpy
* pandas
* sklearn
* matplotlib
* [seaborn](https://seaborn.pydata.org/installing.html)
* [pointpats](https://github.com/pysal/pysal#installation)
* [contextily](https://contextily.readthedocs.io/en/latest/index.html#installation)
* [Folium](https://python-visualization.github.io/folium/installing.html)
* [Plotly](https://plotly.com/python/getting-started/)
* (Jupyter) [notebook](https://jupyter.org/install)

[Anaconda](https://www.anaconda.com/products/distribution) can be installed and the code and the notbooks can be executed in conda environment. For example, numpy, pandas and matplotlib are already inluded in Anaconda.

## Instructions for running

### Machine Learning Pipeling

1. Put original data csv files in resources/data/original
1. Run `python3 src/data_wrangling/join_datasets.py` to join the original 2 datasets together by the NestID
1. Run `python3 src/data_wrangling/clustering.py` to calculate the clusters using DBSCAN with different distances for each year
1. Run `python3 src/data_wrangling/nearby_nests.py` to calculate the closest nest and calculate number of nearby (different distances) neighbours
1. Run `python3 src/data_wrangling/missing_data.py` to impute values to null variables
1. Run `python3 src/data_wrangling/nest_features.py` to compute features related to nests such as the percentage of shy birds in a cluster or in a neighbourhood and also combine the features generated by the previous python scripts together in a single csv file.
1. Run `python3 src/data_wrangling/generate_test_train_split.py` to split the dataset into two parts (test and train) and also generates one hot variables out of some of the features

* Run `python3 src/machine_learning/build_classification_models.py` to build and evaluate classification models
* Run `python3 src/machine_learning/build_clustering_models.py` to build the clustering models which try to find a better propensity threshold
* Run `python3 src/machine_learning/build_regression_models.py` to build and evaluate regression models (these do very poorly)
* Run `python3 src/machine_learning/explaining_models.py` to run MPI and PIMP for feature importance over the decision trees built by the classification script
* Run `python3 src/machine_learning/explaining_model_lime.py` to run LIME for explaining individual predictions

Please note that all scripts must be run from the root repository. For example, if we want to run the clustering phase, we would run `python src/data_wrangling/clustering.py` and not `python3 data_wrangling/clustering.py`

### Jupyter notebooks

Before running the notebooks data files must be created. If not done in machine learning part runt the following commands

* `$ python3 src/data_wrangling/join_datasets.py`
* `$ python src/data_wrangling/clustering.py`
Then go to *visualisation* folder
* `cd src/visualisation/`
* Open Visual Studio Code and navigate there to the location `src/visualisation/` and open the chosen notebook file. Choose the kernel by clicking `Select Kernel` on the right corner and choose one before running the code. If you have installed conda, you may want to choose kernel that name starts with `base`.
* The notebooks can also be executed on the server. Open the notebook server with command `$ jupyter notebook` that opens the folder into a browser and you can choose the wanted notebook. More info can be found [here](https://docs.jupyter.org/en/latest/running.html)
