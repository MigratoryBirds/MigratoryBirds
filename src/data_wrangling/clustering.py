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
CLUSTER_DISTANCES = [15, 30, 50, 100, 200, 300]  # meters

from functools import reduce
import sys
sys.path.append('src/')
from utils.minmax_matrix import MinMaxMatrix
from utils.math_utils import geographic_to_cartesian, euclidean, EARTH_RADIUS
import pandas as pd
import numpy as np


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
                np.array([cluster_id_to_cartesian_1[cluster_id_1]]),
                np.array([cluster_id_to_cartesian_2[cluster_id_2]])
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


if __name__ == '__main__':
    df = pd.read_csv(
        'resources/original_data/FinlandNestDatafile.csv', index_col='NestID'
    )
    df['x'], df['y'], df['z'] \
        = geographic_to_cartesian(df['lat'], df['long'], EARTH_RADIUS)

    resulting_dfs = []
    for cluster_distance in CLUSTER_DISTANCES:
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
                    df_remaining['distance_to_point'] = euclidean(
                        point.values, df_remaining['x y z'.split()].values
                    )
                    nearby_nests = df_remaining[
                        df_remaining['distance_to_point'] < cluster_distance
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
                cluster_id_to_cluster_mean[year][cluster_id] = (
                    get_cluster_mean(df, list(year_to_clusters[year][cluster_id]))
                )

        # build matrix of distances year over year
        this_year_and_next = zip(
            sorted(df['Year'].unique()), sorted(df['Year'].unique()[1:])
        )
        for year, next_year in this_year_and_next:
            (
                distance_matrix,
                index_to_cluster_id_year,
                index_to_cluster_id_next_year
            ) = create_distance_matrix(
                cluster_id_to_cluster_mean[year],
                cluster_id_to_cluster_mean[next_year],
            )
            matrix = MinMaxMatrix(distance_matrix)
            while True:
                min_row, min_col, min_value = matrix.get_min_field()
                if min_value > cluster_distance:
                    break
                (
                    year_to_clusters
                        [next_year]
                        [index_to_cluster_id_year[min_row]]
                ) = (
                    year_to_clusters
                        [next_year]
                        [index_to_cluster_id_next_year[min_col]]
                )
                del (
                    year_to_clusters
                        [next_year]
                        [index_to_cluster_id_next_year[min_col]]
                )
                matrix.soft_delete_row(min_row)
                matrix.soft_delete_column(min_col)

        df_matrix = []
        for year, cluster_id_to_nests in year_to_clusters.items():
            for cluster_id, nests in cluster_id_to_nests.items():
                for nest_id in nests:
                    df_matrix.append([nest_id, cluster_id, year])

        # postprocessing to make the list deterministic
        df_result = pd.DataFrame(
            df_matrix, columns='NestID ClusterID Year'.split()
        ).set_index('NestID')
        df_result['NewClusterID'] = -1
        next_cluster_id = 0
        new_df_matrix = []
        sorted_nest_ids = sorted(df_result.index)
        remaining_nests = set(df_result.index)
        for nest_id in sorted_nest_ids:
            if nest_id not in remaining_nests:
                continue
            subject_cluster_id = df_result.loc[nest_id, 'ClusterID']
            df_result.loc[nest_id, 'NewClusterID'] = next_cluster_id
            remaining_nests.remove(nest_id)
            subject_df = df_result.loc[list(remaining_nests)]
            same_cluster = \
                subject_df[subject_df['ClusterID'] == subject_cluster_id].index
            df_result.loc[same_cluster, 'NewClusterID'] = next_cluster_id
            remaining_nests.difference_update(same_cluster)
            next_cluster_id += 1

        df_result[f'ClusterID_{cluster_distance}'] = df_result['NewClusterID']
        df_result = (
            df_result
            .drop(columns=['NewClusterID'])
            .sort_values([f'ClusterID_{cluster_distance}', 'NestID'])
        )
        df_result['NestID'] = df_result.index
        df_result \
            = df_result[f'NestID Year ClusterID_{cluster_distance}'.split()]
        df_result.reset_index(drop=True, inplace=True)
        resulting_dfs.append(df_result)

    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=['NestID', 'Year']),
        resulting_dfs
    ).to_csv('resources/generated_data/clusters.csv', index=False)
