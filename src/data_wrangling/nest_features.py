"""
Join multiple csv files to get all the information about a nest
into a single big csv file
"""

import sys
sys.path.append('src')
from data_wrangling.join_datasets import data as nests
from data_wrangling.clustering import CLUSTER_DISTANCES
from utils.math_utils import euclidean
import pandas as pd
import numpy as np


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

df = clusters.join(nearby_nests).join(nests.drop('Year', axis=1), how="outer")

print(len(df))

for dist in CLUSTER_DISTANCES:
    df[f'ClusterSize_{dist}'] = [
        sum((df[f'ClusterID_{dist}'] == cid) & (df['Year'] == year))
        for cid, year in zip(df[f'ClusterID_{dist}'], df['Year'])
    ]

# TODO: nearby nests propensity

# ShyBirdsPercentage_Clusters
nest_id_to_shy_birds_percentage = {}
for dist in CLUSTER_DISTANCES:
    for nest_id in df.index:
        cluster_id = df.loc[nest_id, f'ClusterID_{dist}']
        year = df.loc[nest_id, 'Year']
        nests_same_cluster = df[
            (df[f'ClusterID_{dist}'] == cluster_id)
            & (df['Year'] == year)
        ].drop(nest_id)
        if len(nests_same_cluster) == 0:
            nest_id_to_shy_birds_percentage[nest_id] = 0.5
        else:
            total = len(nests_same_cluster)
            shy = sum(nests_same_cluster['Propensity'] == 0)
            aggressive = sum(nests_same_cluster['Propensity'] == 1)
            nulls = total - shy - aggressive
            if nulls == total:
                nest_id_to_shy_birds_percentage[nest_id] = 0.5
            else:
                nest_id_to_shy_birds_percentage[nest_id] = (
                    shy / (shy + aggressive)
                    # (shy + nulls / 2) / (shy + aggressive + nulls)
                )
    df[f'ShyBirdsPercentage_Clusters_{dist}'] = [
        nest_id_to_shy_birds_percentage[k] for k in df.index
    ]

# ShyBirdsPercentage_Nearby
nest_id_to_shy_birds_percentage = {}
for dist in CLUSTER_DISTANCES:
    for nest_id, row in df.iterrows():
        year = df.loc[nest_id, 'Year']
        nests_same_year = df[df['Year'] == df['Year']].drop(nest_id)
        distances = euclidean(
            row['x y z'.split()].values,
            nests_same_year['x y z'.split()].values
        )
        nearby_nests_indices = np.where(distances <= dist)
        nearby_nests = nests_same_year.iloc[nearby_nests_indices]
        if len(nearby_nests) == 0:
            nest_id_to_shy_birds_percentage[nest_id] = 0.5
        else:
            total = len(nearby_nests)
            shy = sum(nearby_nests['Propensity'] == 0)
            aggressive = sum(nearby_nests['Propensity'] == 1)
            nulls = total - shy - aggressive
            if nulls == total:
                nest_id_to_shy_birds_percentage[nest_id] = 0.5
            else:
                nest_id_to_shy_birds_percentage[nest_id] = (
                    shy / (shy + aggressive)
                    # (shy + nulls / 2) / (shy + aggressive + nulls)
                )
    df[f'ShyBirdsPercentage_Nearby_{dist}'] = [
        nest_id_to_shy_birds_percentage[k] for k in df.index
    ]



df = df.dropna(axis=0, subset=['Propensity'])

df = df.join(map_features)
df.to_csv('resources/generated_data/nest_features.csv')
