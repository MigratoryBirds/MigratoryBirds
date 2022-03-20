import pandas as pd

def fetch_data():
    df_general_data = pd.read_csv(
    'resources/generated_data/joined_dataset.csv', index_col='NestID'
    )
    df_clusters = pd.read_csv(
    'resources/generated_data/clusters.csv', index_col='NestID'
    )
    df_location_data = pd.read_csv( 
    'resources/original_data/FinlandNestDatafile.csv', index_col='NestID'
    )
    df_location_data21 = pd.read_csv( 
    'resources/original_data/Finland_nestdata2021_mod.csv', index_col='NestID'
    )
    df_location_data21['Year'] = 2021
    df_location_data = pd.concat([df_location_data, df_location_data21])
    return df_general_data, df_clusters, df_location_data