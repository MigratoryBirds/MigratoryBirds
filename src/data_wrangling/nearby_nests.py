"""
This script generates a csv with information regarding nearby nests
for each nest, as well as the distance to the closest nest.
"""

import sys
sys.path.append('src')
from utils.math_utils import geographic_to_cartesian, euclidean, EARTH_RADIUS
import pandas as pd
import numpy as np

df = pd.read_csv(
    'resources/original_data/FinlandNestDatafile.csv', index_col='NestID'
)['lat long'.split()]
df['x'], df['y'], df['z'] \
    = geographic_to_cartesian(df['lat'], df['long'], EARTH_RADIUS)

nearby_distances = [15, 30, 50, 100, 200, 300]
for dist in nearby_distances:
    df[f'nests_nearby_{dist}'] = 0
df['closest_nest_distance'] = 0
for row in df.iterrows():
    index = row[0]
    row = row[1]
    df_without_nest = df.drop(index)
    distances = euclidean(
        row['x y z'.split()].values, df_without_nest['x y z'.split()].values
    )
    for dist in nearby_distances:
        count = np.sum(distances < dist)
        df.loc[index, f'nests_nearby_{dist}'] = count
    df.loc[index, 'closest_nest_distance'] = np.min(distances)

(
    df
    .drop(columns='lat long'.split())
    .to_csv('resources/generated_data/nearby_nests.csv')
)
