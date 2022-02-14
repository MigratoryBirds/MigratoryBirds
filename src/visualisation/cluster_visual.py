"""
  This python script is mainly used to create different kind of visualisations of the data.
  Both maps and plots are created.
"""
import folium
import pandas as pd
import numpy as np
import plotly.express as px

df_general_data = pd.read_csv(
  'resources/generated_data/joined_dataset.csv', index_col='NestID'
)
df_clusters = pd.read_csv(
  'resources/generated_data/clusters.csv', index_col='NestID'
)
df_location_data = pd.read_csv( 
  'resources/original_data/FinlandNestDatafile.csv', index_col='NestID'
)

def generate_nest_info(general_data: pd.DataFrame, nests: pd.DataFrame) -> list:
    return ([{'ID': nestID, 'Rasps': general_data.loc[nestID]['Rasps'], 
        'Bill_snaps': general_data.loc[nestID]['Bill_snaps'],
        'SnapsRasps':general_data.loc[nestID]['SnapsRasps'], 
        'Propensity':general_data.loc[nestID]['Propensity']} 
        if nestID in general_data.index else nestID for nestID in nests.index.values])

def create_circleMarker(year: int, number_nests: int, df_nests: pd.DataFrame,nest_info:list, color:str, map: folium.Map):
    folium.CircleMarker(
    radius=3*number_nests,
    location=[np.mean(df_nests['lat']), np.mean(df_nests['long'])],
    popup=f"Cluster {i}, Nests: {nest_info}, Year: {year}",
    color=color,
    fill=True,
    fill_color=color,
    opacity=0.5,
    fill_opacity=0.25
).add_to(map)

def save_html_file(file_name: str, figure):
    figure.write_html(f'resources/visualisations/{file_name}', auto_open=False)

# 5 colors for five years
COLORS5 = ['#e00d06','#063de0','#dd1cb7','#580f70','#313695']

# Basic cluster map with folium
lat_coord = (max(df_location_data['lat']) + min(df_location_data['lat']))/2
long_coord = (max(df_location_data['long']) + min(df_location_data['long']))/2
cluster_dist= [15,30,50,100,200,300]
for dist in cluster_dist:
  map = folium.Map(location=[lat_coord, long_coord], default_zoom_start=15)
  year_color_index = 0
  for year in df_clusters['Year'].unique():
      nests_current_year = df_clusters[df_clusters['Year'] == year]
      for i in range(min(df_clusters[f'ClusterID_{dist}']),max(df_clusters[f'ClusterID_{dist}'])+1):
          nests = nests_current_year[nests_current_year[f'ClusterID_{dist}'] == i]
          if len(nests) > 0:
              df_nests = df_location_data.loc[nests.index]
              nest_info = generate_nest_info(df_general_data, nests) 
              create_circleMarker(year, len(nests), df_nests, nest_info, COLORS5[year_color_index], map) 
      year_color_index += 1
  map.save(f'resources/visualisations/clustermap_{dist}.html')
    

# Info plots with multiple column values and plot with SnapsRasps data by Date_nest_found column
df_info = df_general_data.drop(columns=['Year', 'Model', 'Laydate_first_egg',
                      'Site', 'Days_from_LD','Rebuild_original', 'lat', 'long'])
df_info_and_clusters = (pd.merge(df_info, df_clusters, left_on='NestID', right_on='NestID')
    .sort_values(by=['Year','Date_nest_found']))

df_15 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_50','ClusterID_100','ClusterID_200','ClusterID_300'])
df_30 = df_info_and_clusters.drop(columns=['ClusterID_15','ClusterID_50','ClusterID_100','ClusterID_200','ClusterID_300'])
df_50 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_15','ClusterID_100','ClusterID_200','ClusterID_300'])
df_100 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_50','ClusterID_15','ClusterID_200','ClusterID_300'])
df_200 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_50','ClusterID_100','ClusterID_15','ClusterID_300'])
df_300 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_50','ClusterID_100','ClusterID_200','ClusterID_15'])
cluster_dfs = [df_15,df_30,df_50,df_100,df_200,df_300]
index = 0
for df in cluster_dfs:
    fig = px.scatter(x=df['Date_nest_found'], y= df[f'ClusterID_{cluster_dist[index]}'], color=df[f'ClusterID_{cluster_dist[index]}'])
    fig.update_layout(title='Clusters',
                  xaxis_title='Year',
                  yaxis_title='Cluster IDs',
                  font=dict(size=25))
    fig.update_xaxes(rangeslider_visible=True)
    save_html_file(f'clusters_day_found_{cluster_dist[index]}.html', fig)

    fig = px.scatter(x=df['Date_nest_found'], y= df['SnapsRasps'], color=df[f'ClusterID_{cluster_dist[index]}'])
    fig.update_layout(title='SnapsRasps counted in each nest',
                  xaxis_title='Date when nest found',
                  yaxis_title='Number of SnapsRasps',
                  font=dict(size=25))
    fig.update_xaxes(rangeslider_visible=True)
    save_html_file(f'cluster_SnapsRasps_with_range_{cluster_dist[index]}.html', fig)

    df = df.drop(columns=['Date_trial','Date_nest_found'])
    df.index = np.arange(len(df['Rasps']))

    fig = px.parallel_coordinates(data_frame=df, color=df[f'ClusterID_{cluster_dist[index]}'], color_continuous_scale=px.colors.diverging.Earth)
    save_html_file(f'cluster_data_plot_{cluster_dist[index]}.html', fig)

    df = df.drop(columns=['Bill_snaps', 'Rasps'])
    fig = px.parallel_coordinates(data_frame=df,  color=df[f'ClusterID_{cluster_dist[index]}'], color_continuous_scale=px.colors.diverging.Earth)
    save_html_file(f'cluster_data_plot2_{cluster_dist[index]}.html', fig)

    # Create dataframe for clustermaps with and without animation.
    clusterId = []
    cluster_year = []
    cluster_size = []
    lat_centroid = []
    long_centroid = []
    year_color_index = 0
    for year in df_clusters['Year'].unique():
        nests_current_year = df_clusters[df_clusters['Year'] == year]
        for i in range(min(df_clusters[f'ClusterID_{cluster_dist[index]}']),max(df_clusters[f'ClusterID_{cluster_dist[index]}'])+1):
            nests = nests_current_year[nests_current_year[f'ClusterID_{cluster_dist[index]}'] == i]
            if len(nests) > 0:
              df_nests = df_location_data.loc[nests.index]
              clusterId.append(i)
              cluster_size.append(len(nests))
              cluster_year.append(year)
              lat_centroid.append(np.mean(df_nests['lat']))
              long_centroid.append(np.mean(df_nests['long']))
    df3 = pd.DataFrame(
                {f'ClusterID_{cluster_dist[index]}': clusterId,'Cluster_size': cluster_size, 'Year': cluster_year,
                'Lat_centroid': lat_centroid, 'Long_centroid':long_centroid})

    # Cluster map with animation
    fig = px.scatter_mapbox(df3, center={'lat': lat_coord , 'lon': long_coord}, lat='Lat_centroid', lon='Long_centroid',
        color=f'ClusterID_{cluster_dist[index]}', size='Cluster_size',
        color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=10, animation_frame='Year',
        animation_group=f'ClusterID_{cluster_dist[index]}',hover_data=[f'ClusterID_{cluster_dist[index]}', 'Year'], mapbox_style='carto-positron')
    fig['layout'].pop('updatemenus')
    save_html_file(f'clustermap_with_animation_{cluster_dist[index]}.html', fig)

    # Cluster map without animation
    fig = px.scatter_mapbox(df3, center={'lat': lat_coord , 'lon': long_coord}, lat='Lat_centroid', lon='Long_centroid',
        color=f'ClusterID_{cluster_dist[index]}', size='Cluster_size',
        color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=10,
        animation_group=f'ClusterID_{cluster_dist[index]}',hover_data=[f'ClusterID_{cluster_dist[index]}', 'Year'], mapbox_style='carto-positron')
    save_html_file(f'clustermap_without_animation_{cluster_dist[index]}.html', fig)
    index += 1