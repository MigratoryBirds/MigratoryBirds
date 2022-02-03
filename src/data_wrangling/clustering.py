"""
This script clusters the bird nests together based on their euclidean
distances from one another.

We use a custom clustering algorithm where we start from one point and
cluster all the points next to it together, then add the points which are
next to all the points (recursively) to the same cluster.

Source for converting longitude and latitude to x, y, z coordinates:
https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
"""

# constants
NEARBY_DISTANCE = 15  # meters
EARTH_RADIUS = 6378000  # meters

import sys
sys.path.append('src/')
from utils.minmax_matrix import MinMaxMatrix
import pandas as pd
import numpy as np

df = pd.read_csv(
    'resources/original_data/FinlandNestDatafile.csv', index_col='NestID'
)


def add_cartesian_to_geographic(
    df: pd.DataFrame, earth_radius: float
) -> pd.DataFrame:
    cos_lat = np.cos(df['lat'].map(np.radians))
    cos_lon = np.cos(df['long'].map(np.radians))
    sin_lat = np.sin(df['long'].map(np.radians))
    sin_lon = np.sin(df['lat'].map(np.radians))
    df['x'] = earth_radius * cos_lat * cos_lon
    df['y'] = earth_radius * cos_lat * sin_lon
    df['z'] = earth_radius * sin_lat
    return df


def euclidean(p1: tuple, p2: tuple) -> float:
    return np.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(p1, p2)]))


def get_cluster_mean(
    df: pd.DataFrame, nest_ids: list[str]
) -> tuple[float]:
    values = np.hstack([
        df.loc[nest_ids, 'x y z'.split()]
    ])
    return values.mean(axis=0)


def create_distance_matrix(
    cluster_id_to_cartesian_1: dict[int, tuple[float]],
    cluster_id_to_cartesian_2: dict[int, tuple[float]]
) -> list[list[float]]:
    distance_matrix = []
    cluster_ids_1 = list(cluster_id_to_cartesian_1.keys())
    cluster_ids_2 = list(cluster_id_to_cartesian_2.keys())
    for cluster_id_1 in cluster_ids_1:
        distance_list = [
            euclidean(
                cluster_id_to_cartesian_1[cluster_id_1],
                cluster_id_to_cartesian_2[cluster_id_2]
            )
            for cluster_id_2 in cluster_ids_2
        ]
        distance_matrix.append(distance_list)
    cluster_id_to_index_1 = {
        i: cluster_id for i, cluster_id in enumerate(cluster_ids_1)
    }
    cluster_id_to_index_2 = {
        i: cluster_id for i, cluster_id in enumerate(cluster_ids_2)
    }
    return distance_matrix, cluster_id_to_index_1, cluster_id_to_index_2


df = add_cartesian_to_geographic(df, EARTH_RADIUS)

next_cluster_id = 0
year_to_clusters = {}
for year in sorted(df['Year'].unique()):
    year_df = df[df['Year'] == year]
    remaining_nests = set(year_df.index)
    cluster_id_to_nests = {}
    while len(remaining_nests) > 0:
        # start new cluster
        current_cluster = set()
        cluster_queue = {remaining_nests.pop()}
        while len(cluster_queue) > 0:
            nest_id = cluster_queue.pop()
            current_cluster.add(nest_id)
            point = year_df.loc[nest_id]['x y z'.split()]
            df_remaining = year_df.loc[list(remaining_nests)]
            if len(df_remaining) == 0:
                continue
            # calculate distances
            df_remaining['distance_to_point'] = df_remaining.apply(
                lambda x: euclidean(point, x['x y z'.split()]),
                axis=1
            )
            nearby_nests = df_remaining[
                df_remaining['distance_to_point'] < NEARBY_DISTANCE
            ].index
            cluster_queue.update(nearby_nests)
            remaining_nests.difference_update(nearby_nests)
        cluster_id_to_nests[next_cluster_id] = current_cluster
        next_cluster_id += 1
    year_to_clusters[year] = cluster_id_to_nests

cluster_id_to_cluster_mean = {}
for year in sorted(df['Year'].unique()):
    cluster_id_to_cluster_mean[year] = {}
    for cluster_id in year_to_clusters[year]:
        cluster_id_to_cluster_mean[year][cluster_id] = \
            get_cluster_mean(df, list(year_to_clusters[year][cluster_id]))


# build matrix of distances year over year
this_year_and_next = zip(
    sorted(df['Year'].unique()), sorted(df['Year'].unique()[1:])
)
for year, next_year in this_year_and_next:
    distance_matrix, index_to_cluster_id_year, index_to_cluster_id_next_year = \
        create_distance_matrix(
            cluster_id_to_cluster_mean[year],
            cluster_id_to_cluster_mean[next_year],
        )
    matrix = MinMaxMatrix(distance_matrix)
    while True:
        min_row, min_col, min_value = matrix.get_min_field()
        if min_value > NEARBY_DISTANCE:
            break
        year_to_clusters[next_year][index_to_cluster_id_year[min_row]] \
            = year_to_clusters[
                next_year
            ][index_to_cluster_id_next_year[min_col]]
        del year_to_clusters[next_year][index_to_cluster_id_next_year[min_col]]
        matrix.soft_delete_row(min_row)
        matrix.soft_delete_column(min_col)

df_matrix = []
for year, cluster_id_to_nests in year_to_clusters.items():
    for cluster_id, nests in cluster_id_to_nests.items():
        for nest_id in nests:
            df_matrix.append([nest_id, cluster_id, year])

# postprocessing to make the list deterministic
df = pd.DataFrame(df_matrix, columns='NestID ClusterID Year'.split())
df = df.set_index('NestID')
df['NewClusterID'] = -1
next_cluster_id = 0
new_df_matrix = []
sorted_nest_ids = sorted(df.index)
remaining_nests = set(df.index)
for nest_id in sorted_nest_ids:
    if nest_id not in remaining_nests:
        continue
    subject_cluster_id = df.loc[nest_id, 'ClusterID']
    df.loc[nest_id, 'NewClusterID'] = next_cluster_id
    remaining_nests.remove(nest_id)
    subject_df = df.loc[list(remaining_nests)]
    same_cluster = \
        subject_df[subject_df['ClusterID'] == subject_cluster_id].index
    df.loc[same_cluster, 'NewClusterID'] = next_cluster_id
    remaining_nests.difference_update(same_cluster)
    next_cluster_id += 1

df['ClusterID'] = df['NewClusterID']
df = df.drop(columns=['NewClusterID'])
df = df.sort_values(['ClusterID', 'NestID'])
df['NestID'] = df.index
df = df['NestID ClusterID Year'.split()]
df.to_csv('resources/generated_data/clusters.csv', index=False)
