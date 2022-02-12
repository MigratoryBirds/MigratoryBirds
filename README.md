# MigratoryBirds
Project for DSP2022 - Data Science Project I - Detecting micro and macro spatial-temporal patterns in behaviour and habitat selection by a migratory bird

## Installation Instructions

* You must have python3.9 or higher installed


## Instructions for running

1. Put data csv files in resources/data/original
1. Run data\_wrangling/clustering.py

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

Data must be generated first by running data\_wrangling.py in code/data\_wrangling directory.

// Needs still modifications that one can run it both with unix and windows.

`code/data\_wrangling directory$ python3 run data\_wrangling.py`

Move then back to root of the repository.

`code/data\_wrangling directory$ cd ../..`

Create clustering data

`$ python src/data_wrangling/clustering.py`

Finally create visualisations with attribute `unix` or `windows` depending on your operation system

`$ python3 src/visualisation/cluster_visual.py <your_os>`

// Should be still tested for windows files
