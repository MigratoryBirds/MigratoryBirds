"""
  This python script is mainly used to create different kind of visualisations of the data.
  Both maps and plots are created.
"""
import folium
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

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

def generate_nest_info(general_data: pd.DataFrame, nests: pd.DataFrame) -> list:
    return ([{'ID': nestID, 'Rasps': general_data.loc[nestID]['Rasps'], 
        'Bill_snaps': general_data.loc[nestID]['Bill_snaps'],
        'SnapsRasps':general_data.loc[nestID]['SnapsRasps'], 
        'Propensity':general_data.loc[nestID]['Propensity']} 
        if nestID in general_data.index else nestID for nestID in nests.index.values])

def generate_cluster_info(cluster: int, year: int, general_data: pd.DataFrame, 
    nests: pd.DataFrame, cluster_size_data: pd.DataFrame, dist:int):
  nests_without_data = len([nestID for nestID in nests.index.values])
  nests_with_data = len([nestID for nestID in nests.index.values if nestID in general_data.index])
  sites = list(set([df_location_data.loc[nestID]['Site'].strip() for nestID in nests.index.values]))
  total_angrybirds = sum([general_data.loc[nestID]['Propensity'] for nestID in nests.index.values if nestID in general_data.index])
  row = pd.DataFrame({'ClusterID': [cluster], 'Year': [year], 'Dist': [dist], 'Site': [sites], 'Nests_with_data': [nests_with_data],
     'All_nests': [nests_without_data], 'Total_angrybirds': [total_angrybirds]})
  return cluster_size_data.append(row, ignore_index=True)

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
cluster_size_data  = pd.DataFrame(columns=['ClusterID', 'Year', 'Dist', 'Site',
    'Nests_with_data','All_nests','Total_angrybirds'])

for dist in cluster_dist:
    map = folium.Map(location=[lat_coord, long_coord], default_zoom_start=15)
    year_color_index = 0
    years = df_clusters['Year'].unique()
    for year in years:
        nests_current_year = df_clusters[df_clusters['Year'] == year]
        for i in range(min(df_clusters[f'ClusterID_{dist}']),max(df_clusters[f'ClusterID_{dist}'])+1):
            nests = nests_current_year[nests_current_year[f'ClusterID_{dist}'] == i]
            if len(nests) > 0:
                df_nests = df_location_data.loc[nests.index]
                nest_info = generate_nest_info(df_general_data, nests)
                cluster_size_data = generate_cluster_info(i, year, df_general_data, nests, cluster_size_data, dist)
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
    df = df.drop(columns=['Date_trial','Date_nest_found', 'New_rebuild'])
    df.index = np.arange(len(df['Rasps']))
    df = df.drop(columns=['Bill_snaps', 'Rasps'])
    fig = px.parallel_coordinates(data_frame=df,  color=df[f'ClusterID_{cluster_dist[index]}'], color_continuous_scale=px.colors.diverging.Earth)
    save_html_file(f'cluster_data_plot_{cluster_dist[index]}.html', fig)

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
    index += 1

year = df_location_data['Year'].unique()
cluster_data_by_site = pd.DataFrame(columns=['Year', 'Site', 'Total_angrybirds', 'All_nests'])
for year in years:
    for site in df_location_data['Site'].str.strip().unique():
        attacks = sum([np.sum(df_general_data[(df_general_data['Site']==site2)&(df_general_data['Year']==year)]['Propensity']) if site == site2 else 0 for site2 in df_general_data['Site'].str.strip().unique()])
        size_all = len([nestID for nestID in df_location_data[(df_location_data['Site'] == site)&(df_location_data['Year'] == year)].index.values])
        row = pd.DataFrame({'Site': [site], 'Year': [year], 'Total_angrybirds': [attacks], 'All_nests': [size_all]})
        cluster_data_by_site = cluster_data_by_site.append(row, ignore_index=True)

cluster_data_by_site.to_csv("resources/generated_data/cluster_data.csv", index=False)
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=20)
for year in years:
    fig, ax1 = plt.subplots()
    #plt.title('Attacks and Sizes of the Cluaters', fontsize=20)
    ax1.set_xlabel('ClusterID')
    ax1.set_ylabel('Size of the Cluster',color = 'blue')
    ax1.scatter(cluster_size_data[(cluster_size_data['Dist']==300) & (cluster_size_data['Year']==year)]['ClusterID'], 
        cluster_size_data[(cluster_size_data['Dist']==300) & (cluster_size_data['Year']==year)]['All_nests'], color='blue')
    ax1.tick_params(axis ='y', labelcolor = 'blue') 
    ax2 = ax1.twinx() 
      
    ax2.set_ylabel('Number of attacks', color = 'red')
    ax2.scatter(cluster_size_data[(cluster_size_data['Dist']==300) & (cluster_size_data['Year']==year)]['ClusterID'], 
        cluster_size_data[(cluster_size_data['Dist']==300) & (cluster_size_data['Year']==year)]['Total_angrybirds'],
        color='red')
    ax2.tick_params(axis ='y', labelcolor = 'red')

    plt.savefig(f'resources/visualisations/cluster_prospensity{year}.png')

    fig, ax1 = plt.subplots(figsize=(18,18))
    #plt.title('Attacks and Sizes of the Sites', fontsize=20)
    ax1.set_ylabel('Size of the Cluster',color = 'blue')
    x=cluster_data_by_site[(cluster_data_by_site['Year']==year)]['Site']
    ax1.scatter(x, 
        cluster_data_by_site[(cluster_data_by_site['Year']==year)]['All_nests'], color='blue')
    ax1.tick_params(axis ='y', labelcolor = 'blue')
    ax1.tick_params(axis ='x',rotation=90)
 
    ax2 = ax1.twinx() 
      
    ax2.set_ylabel('Number of attacks', color = 'red')
    x=cluster_data_by_site[(cluster_data_by_site['Year']==year)]['Site']
    y=cluster_data_by_site[(cluster_data_by_site['Year']==year)]['Total_angrybirds']
    ax2.scatter(x, y,
        color='red')
    ax2.tick_params(axis ='y', labelcolor = 'red')

    plt.savefig(f'resources/visualisations/cluster_prospensity{year}_by_Site.png')