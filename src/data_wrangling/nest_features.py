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
).drop(columns=['Year'])

nearby_nests = pd.read_csv(
    'resources/generated_data/nearby_nests.csv', index_col='NestID'
)

for dist in CLUSTER_DISTANCES:
    clusters[f'ClusterSize_{dist}'] = [
        sum(clusters[f'ClusterID_{dist}'] == cid)
        for cid in clusters[f'ClusterID_{dist}']
    ]

df = nests.join(clusters, how="inner").join(nearby_nests)

df.to_csv('resources/generated_data/nest_features.csv')
