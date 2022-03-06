# MigratoryBirds
Project for DSP2022 - Data Science Project I - Detecting micro and macro spatial-temporal patterns in behaviour and habitat selection by a migratory bird

## Installation Instructions

* You must have python3.9 or higher installed
* [Folium](https://python-visualization.github.io/folium/installing.html) and [Plotly](https://plotly.com/python/getting-started/) must be installed for visualisation.


## Instructions for running

1. Put data csv files in resources/data/original
1. Run `python3 src/data_wrangling/join_datasets.py`
1. Run `python3 src/data_wrangling/clustering.py`
1. Run `python3 src/data_wrangling/nearby_nests.py`
1. Run `python3 src/data_wrangling/missing_data.py`
1. Run `python3 src/data_wrangling/nest_features.py`
1. Run `python3 src/data_wrangling/generate_test_train_split.py`
1. Run `python3 src/data_wrangling/build_multiclass_models_sklearn.py`

Please note that all scripts must be run from the root repository. For example, if we want to run the clustering phase, we would run `python src/data_wrangling/clustering.py`

### Before running the visualisation.py

NB: In some cases you need to use `python3` (or alike) instead of `python`.

1. Generate data
`$ python src/data_wrangling/join_datasets.py`

2. Create clustering data

`$ python src/data_wrangling/clustering.py`

3. Create visualisations

`$ python src/visualisation/cluster_visual.py`
