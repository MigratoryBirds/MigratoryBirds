"""
    This script reads data into different dataframes for map and data visualisation tasks.
    Returns:
    df_general: nesting behaviour data for each years
    df_clusters: nests cluster ids for each cluster with different distances (15,30,50,100,200,300)
    df_location: all nests with their location data
"""
import pandas as pd

def fetch_data(dots=True):
    add_dots = ''
    if dots:
        add_dots ='../../'
    df_general = pd.read_csv(
        f'{add_dots}resources/generated_data/joined_dataset.csv', index_col='NestID'
    )
    df_clusters = pd.read_csv(
        f'{add_dots}resources/generated_data/clusters.csv', index_col='NestID'
    )
    df_location = pd.read_csv( 
        f'{add_dots}resources/original_data/FinlandNestDatafile.csv', index_col='NestID'
    )
    df_location21 = pd.read_csv( 
        f'{add_dots}resources/original_data/Finland_nestdata2021_mod.csv', index_col='NestID'
    )
    df_location21['Year'] = 2021
    df_location = pd.concat([df_location, df_location21])
    return df_general, df_clusters, df_location
