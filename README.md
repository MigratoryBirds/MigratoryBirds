# MigratoryBirds
Project for DSP2022 - Data Science Project I - Detecting micro and macro spatial-temporal patterns in behaviour and habitat selection by a migratory bird

## Installation Instructions

* You must have python3.9 or higher installed


## Instructions for running

1. Put data csv files in resources/data/original
1. Run data\_wrangling/clustering.py
1. Run data\_wrangling/nearby\_nests.py
1. Run data\_wrangling/nest\_features.py
1. Run data\_wrangling/generate\_test\_train\_split.py

Please note that all scripts must be run from the root repository. For example, if we want to run the clustering phase, we would run `python src/data_wrangling/clustering.py`

### Before running the visualisation.py

Libraries folium and plotly must be installed.

Folium can be installed with following commands
```
$ pip3 install folium
```
or
```
$ conda install folium -c conda-forge
```
For plotly installation instructors can be found [here](https://plotly.com/python/getting-started/).

NB: In some cases yuor `python` command might be `python3` or alike.

Data must be generated first by
`$ python src/data_wrangling/join_datasets.py`

Create clustering data

`$ python src/data_wrangling/clustering.py`

Finally create visualisations

`$ python src/visualisation/cluster_visual.py`
