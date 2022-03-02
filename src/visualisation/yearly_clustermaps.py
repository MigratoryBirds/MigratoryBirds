"""
  This python script is mainly used to create different kind of visualisations of the data.
  Both maps and plots are created.
"""
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np

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

YEARS = df_clusters['Year'].unique()
# Basic cluster map with folium
lat_coord = (max(df_location_data['lat']) + min(df_location_data['lat']))/2
long_coord = (max(df_location_data['long']) + min(df_location_data['long']))/2
for dist in [15,50,100,200]:
    for year in YEARS:
        map = folium.Map(location=[lat_coord, long_coord], default_zoom_start=15, control_scale=True)
        cluster = MarkerCluster(name=f'Nests {year}')
        nests_current_year = df_clusters[df_clusters['Year'] == year]
        for i in range(min(df_clusters[f'ClusterID_{dist}']),max(df_clusters[f'ClusterID_{dist}'])+1):
            nests = nests_current_year[nests_current_year[f'ClusterID_{dist}'] == i]
            if len(nests) > 0:
                df_nests = df_location_data.loc[nests.index]
                df_nests['NestID'] = df_nests.index.values
                df_nests['Propensity'] = [df_general_data.loc[nestID]['Propensity'] if nestID in df_general_data.index else np.nan for nestID in nests.index.values]
                df_nests.apply(
                    lambda row:
                        folium.Marker(
                            location=[row['lat'],row['long']],
                            popup= [i,row['NestID'],row['Site'],row['Propensity']],
                            tooltip='<h5>Click for more info</5>',
                        ).add_to(cluster),
                        axis=1)
                cluster.add_to(map)
        map.save(f'resources/visualisations/maps/clustermap_{dist}_{year}.html')