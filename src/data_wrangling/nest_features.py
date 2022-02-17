"""
Join multiple csv files to get all the information about a nest
into a single big csv file
"""

import sys
sys.path.append('src')
from data_wrangling.join_datasets import data as nests
from data_wrangling.clustering import CLUSTER_DISTANCES
import pandas as pd

nests = nests.set_index('NestID')

clusters = pd.read_csv(
    'resources/generated_data/clusters.csv', index_col='NestID'
)

nearby_nests = pd.read_csv(
    'resources/generated_data/nearby_nests.csv', index_col='NestID'
)
map_features = pd.read_csv(
    'resources/generated_data/map_features.csv', index_col='NestID'
)

df = clusters.join(nearby_nests)

for dist in CLUSTER_DISTANCES:
    df[f'ClusterSize_{dist}'] = [
        sum((df[f'ClusterID_{dist}'] == cid) & (df['Year'] == year))
        for cid, year in zip(df[f'ClusterID_{dist}'], df['Year'])
    ]

df = df.join(nests.drop('Year', axis=1), how="inner").join(map_features)
df.to_csv('resources/generated_data/nest_features.csv')
