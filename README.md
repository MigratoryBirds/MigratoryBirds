# MigratoryBirds
Project for DSP2022 - Data Science Project I - Detecting micro and macro spatial-temporal patterns in behaviour and habitat selection by a migratory bird

## Installation Instructions

* You must have python3.9 or higher installed
* [Folium](https://python-visualization.github.io/folium/installing.html) and [Plotly](https://plotly.com/python/getting-started/) must be installed for visualisation.


## Instructions for running

1. Put data csv files in resources/data/original
1. Run data\_wrangling/clustering.py
1. Run data\_wrangling/nearby\_nests.py
1. Run data\_wrangling/nest\_features.py
1. Run data\_wrangling/generate\_test\_train\_split.py
1. Run data\_wrangling/build\_multiclass\_models\_sklearn.py

Please note that all scripts must be run from the root repository. For example, if we want to run the clustering phase, we would run `python src/data_wrangling/clustering.py`

### Before running the visualisation.py

NB: In some cases you need to use `python3` (or alike) instead of `python`.

1. Generate data
`$ python src/data_wrangling/join_datasets.py`

2. Create clustering data

`$ python src/data_wrangling/clustering.py`

3. Create visualisations

`$ python src/visualisation/cluster_visual.py`
