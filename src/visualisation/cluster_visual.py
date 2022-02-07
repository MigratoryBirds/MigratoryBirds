import folium
import pandas as pd
import numpy as np
import sys
import plotly.express as px

# Read the arguments given to command python3. Note that args[0] is the name of the file.
args = sys.argv

if args[1] == 'windows':
    df_general_data = pd.read_csv(
      'resources\\generated_data\\generated_data.csv', index_col='NestID'
    )
    df_clusters = pd.read_csv(
      'resources\\generated_data\\clusters.csv', index_col='NestID'
    )
    df_locations_data = pd.read_csv( 
      'resources\\original_data\\FinlandNestDatafile.csv', index_col='NestID'
    )
elif args[1] == 'unix':
    df_general_data = pd.read_csv(
      'resources/generated_data/generated_data.csv', index_col='NestID'
    )
    df_clusters = pd.read_csv(
      'resources/generated_data/clusters.csv', index_col='NestID'
    )
    df_location_data = pd.read_csv( 
      'resources/original_data/FinlandNestDatafile.csv', index_col='NestID'
    )

def generate_nest_info(general_data: pd.DataFrame, nests: pd.DataFrame) -> list:
    return [{'ID': nestID, 'Rasps': general_data.loc[nestID]['Rasps'], 
        'Bill_snaps': general_data.loc[nestID]['Bill_snaps'],
        'SnapsRasps':general_data.loc[nestID]['SnapsRasps'], 
        'Propensity':general_data.loc[nestID]['Propensity']} 
        if nestID in general_data.index else nestID for nestID in nests.index.values]

def create_circleMarker(number_nests: int, df_nests: pd.DataFrame,nest_info:list, color:str, map: folium.Map):
    folium.CircleMarker(
    radius=3*number_nests,
    location=[np.mean(df_nests['lat']), np.mean(df_nests['long'])],
    popup=f"Cluster {i}, Nests: {nest_info}, Year: 2020",
    color=color,
    fill=True,
    fill_color=color,
    opacity=0.5,
    fill_opacity=0.25
).add_to(map)

def save_html_file(os_t: str, file_name: str, figure):
  if os_t == 'unix':
      figure.write_html(f'resources/visualisations/{file_name}', auto_open=False)
  elif os_t == 'windows':
      figure.write_html(f'resources\\visualisations\\{file_name}', auto_open=False)

df_clusters2019 = df_clusters[df_clusters['Year'] == 2019]
df_clusters2020 = df_clusters[df_clusters['Year'] == 2020]

COLOR2019 = '#0477c9'
COLOR2020 = '#e50050'

# Basic cluster map with folium
lat_coord = (max(df_location_data['lat']) + min(df_location_data['lat']))/2
long_coord = (max(df_location_data['long']) + min(df_location_data['long']))/2
map = folium.Map(location=[lat_coord, long_coord], default_zoom_start=15)
for i in range(min(df_clusters['ClusterID']),max(df_clusters['ClusterID'])+1):
    # 2019
    nests2019 = df_clusters2019[df_clusters2019['ClusterID'] == i]
    if len(nests2019) > 0:
        df_nests = df_location_data.loc[nests2019.index]
        nest_info = generate_nest_info(df_general_data, nests2019) 
        create_circleMarker(len(nests2019), df_nests, nest_info, COLOR2019 , map) 
    # 2020
    nests2020 = df_clusters2020[df_clusters2020['ClusterID'] == i]
    if len(nests2020) > 0:
        df_nests = df_location_data.loc[nests2020.index]
        nest_info = generate_nest_info(df_general_data, nests2020)
        create_circleMarker(len(nests2020), df_nests, nest_info, COLOR2020, map) 
if args[1] == 'unix':
    map.save('resources/visualisations/clustermap.html')
elif args[1] == 'windows':
    map.save('resources\\visualisations\\clustermap.html')

# Info plots with multiple column values and plot with SnapsRasps data by Date_nest_found column
df_info = df_general_data.drop(columns=['Year', 'Model', 'Laydate_first_egg',
                      'Site', 'Days_from_LD','Rebuild_original', 'lat', 'long', 'Unnamed: 0'])
df_info_and_clusters = pd.merge(df_info, df_clusters, left_on='NestID', right_on='NestID')

df = df_info_and_clusters.sort_values(by=['Year','Date_nest_found'])

fig = px.scatter(x=df['Date_nest_found'], y= df['ClusterID'], color=df['ClusterID'])
fig.update_layout(title='Clusters',
              xaxis_title='Year',
              yaxis_title='Cluster IDs',
              font=dict(size=25))
fig.update_xaxes(rangeslider_visible=True)
save_html_file(args[1], 'clusters_day_found.html', fig)

fig = px.scatter(x=df['Date_nest_found'], y= df['SnapsRasps'], color=df['ClusterID'])
fig.update_layout(title='SnapsRasps counted in each nest',
              xaxis_title='Date when nest found',
              yaxis_title='Number of SnapsRasps',
              font=dict(size=25))
fig.update_xaxes(rangeslider_visible=True)
save_html_file(args[1], 'cluster_SnapsRasps_with_range.html', fig)
df = df.drop(columns=['Date_trial','Date_nest_found'])

fig = px.parallel_coordinates(df,  color='ClusterID', labels={'ClusterID': 'Cluster ID', 'Propensity': 'Propensity',
                              'Rasps': 'Rasps', 'Bill_snaps': 'Bill Snaps','SnapsRasps': 'SnapsRasps',
                              'Cuckoo_perch': 'Cuckoo Perch', 'Year': 'Year' 
                              }, color_continuous_midpoint=3, color_continuous_scale=px.colors.diverging.Tealrose)
save_html_file(args[1], 'clusterdata.html', fig)

df = df.drop(columns=['Bill_snaps', 'Rasps'])
fig = px.parallel_coordinates(df,  color='ClusterID', labels={'ClusterID': 'Cluster ID', 'Propensity': 'Propensity',
                    'SnapsRasps': 'SnapsRasps', 'Cuckoo_perch': 'Cuckoo Perch', 'Year': 'Year' 
                    }, color_continuous_midpoint=3, color_continuous_scale=px.colors.diverging.Tealrose)
save_html_file(args[1], 'clusterdata2.html', fig)

# Create dataframe for clustermaps with and without animation.
clusterId = []
year = []
cluster_size = []
lat_centroid = []
long_centroid = []
for i in range(min(df_clusters['ClusterID']),max(df_clusters['ClusterID'])+1):
  # 2019
  nests2019 = df_clusters2019[df_clusters2019['ClusterID'] == i]
  if len(nests2019) > 0:
    df_nests = df_location_data.loc[nests2019.index]
    clusterId.append(i)
    cluster_size.append(len(nests2019))
    year.append(2019)
    lat_centroid.append(np.mean(df_nests['lat']))
    long_centroid.append(np.mean(df_nests['long']))
  # 2020
  nests2020 = df_clusters2020[df_clusters2020['ClusterID'] == i]
  if len(nests2020) > 0:
    df_nests = df_location_data.loc[nests2020.index]
    clusterId.append(i)
    cluster_size.append(len(nests2020))
    year.append(2020)
    lat_centroid.append(np.mean(df_nests['lat']))
    long_centroid.append(np.mean(df_nests['long']))

df3 = pd.DataFrame(
            {'ClusterID': clusterId,'Cluster_size': cluster_size, 'Year': year,
             'Lat_centroid': lat_centroid, 'Long_centroid':long_centroid})

# Cluster map with animation
fig = px.scatter_mapbox(df3, center={'lat': lat_coord , 'lon': long_coord}, lat='Lat_centroid', lon='Long_centroid', color='ClusterID', size='Cluster_size',
    color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=10, animation_frame='Year',
    animation_group='ClusterID',hover_data=['ClusterID', 'Year'], mapbox_style='carto-positron')
fig['layout'].pop('updatemenus')
save_html_file(args[1], 'clustermap_with_animation.html', fig)

# Cluster map without animation
fig = px.scatter_mapbox(df3, center={'lat': lat_coord , 'lon': long_coord}, lat='Lat_centroid', lon='Long_centroid', color='ClusterID', size='Cluster_size',
    color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, zoom=10,
    animation_group='ClusterID',hover_data=['ClusterID', 'Year'], mapbox_style='carto-positron')
save_html_file(args[1], 'clustermap_without_animation.html', fig)