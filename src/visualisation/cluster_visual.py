"""
  This python script is mainly used to create different kind of visualisations of the data.
  Both maps and plots are created.
"""
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

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

def generate_cluster_info(cluster: int, year: int, general_data: pd.DataFrame, 
    nests: pd.DataFrame, cluster_size_data: pd.DataFrame, dist:int):
    all_nests = len([nestID for nestID in nests.index.values])
    nests_with_data = len([nestID for nestID in nests.index.values if nestID in general_data.index])
    sites = list(set([df_location_data.loc[nestID]['Site'].strip() for nestID in nests.index.values]))
    total_propensity = -1*sum([general_data.loc[nestID]['Propensity'] for nestID in nests.index.values if nestID in general_data.index])
    row = pd.DataFrame({'ClusterID': [f'ClusterID {cluster}'], 'Year': [year], 'Dist': [dist], 'Site': [sites], 'Nests_with_data': [nests_with_data],
        'All_nests': [all_nests], 'Total_propensity': [total_propensity]})
    return cluster_size_data.append(row, ignore_index=True)

def save_html_file(file_name: str, figure):
    figure.write_html(f'resources/visualisations/{file_name}', auto_open=False)

YEARS = df_clusters['Year'].unique()
# Basic cluster map with folium
lat_coord = (max(df_location_data['lat']) + min(df_location_data['lat']))/2
long_coord = (max(df_location_data['long']) + min(df_location_data['long']))/2
for dist in [15,50,200]:
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
        map.save(f'resources/visualisations/clustermap_{dist}_{year}.html')

# Info plots with multiple column values and plot with SnapsRasps data by Date_nest_found column
df_info = df_general_data.drop(columns=['Year', 'Model', 'Laydate_first_egg',
                      'Site', 'Days_from_LD','Rebuild_original', 'lat', 'long'])
df_info_and_clusters = (pd.merge(df_info, df_clusters, left_on='NestID', right_on='NestID')
    .sort_values(by=['Year','Date_nest_found']))


df_15 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_50','ClusterID_100','ClusterID_200'])
df_30 = df_info_and_clusters.drop(columns=['ClusterID_15','ClusterID_50','ClusterID_100','ClusterID_200'])
df_50 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_15','ClusterID_100','ClusterID_200'])
df_100 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_50','ClusterID_15','ClusterID_200'])
df_200 = df_info_and_clusters.drop(columns=['ClusterID_30','ClusterID_50','ClusterID_100','ClusterID_15'])
cluster_dfs = [df_15,df_30,df_50,df_100,df_200]
index = 0
cluster_dist= [15,30,50,100, 200]
for df in cluster_dfs:
    df = df.drop(columns=['Date_trial','Date_nest_found', 'New_rebuild'])
    df.index = np.arange(len(df['Rasps']))
    df = df.drop(columns=['Bill_snaps', 'Rasps'])
    fig = px.parallel_coordinates(data_frame=df,  color=df[f'ClusterID_{cluster_dist[index]}'], color_continuous_scale=px.colors.diverging.Earth)
    save_html_file(f'cluster_data_plot_{cluster_dist[index]}.html', fig)
    index += 1

# Create site size vs propensity data
cluster_data_by_site = pd.DataFrame(columns=['Year', 'Site', 'Total_angrybirds', 'All_nests'])
for year in YEARS:
    for site in df_location_data['Site'].str.strip().unique():
        total_propensity = -1*sum([np.sum(df_general_data[(df_general_data['Site']==site2)&(df_general_data['Year']==year)]['Propensity']) if site == site2 else 0 for site2 in df_general_data['Site'].str.strip().unique()])
        size_all = len([nestID for nestID in df_location_data[(df_location_data['Site'] == site)&(df_location_data['Year'] == year)].index.values])
        row = pd.DataFrame({'Site': [site], 'Year': [year], 'Total_propensity': [total_propensity], 'All_nests': [size_all]})
        cluster_data_by_site = cluster_data_by_site.append(row, ignore_index=True)

# cluster_data_by_site.to_csv("resources/generated_data/cluster_data.csv", index=False)
# Create cluster size vs propensity data
cluster_size_data  = pd.DataFrame(columns=['ClusterID', 'Year', 'Dist', 'Site',
    'Nests_with_data','All_nests','Total_angrybirds'])
cluster_size_data = cluster_size_data.astype({'ClusterID': str})
for dist in cluster_dist:
    for year in YEARS:
        nests_current_year = df_clusters[df_clusters['Year'] == year]
        for i in range(min(df_clusters[f'ClusterID_{dist}']),max(df_clusters[f'ClusterID_{dist}'])+1):
            nests = nests_current_year[nests_current_year[f'ClusterID_{dist}'] == i]
            if len(nests) > 0:
                cluster_size_data = generate_cluster_info(i, year, df_general_data, nests, cluster_size_data, dist)
for year in YEARS:
    df = cluster_size_data[(cluster_size_data.Dist==200) & (cluster_size_data.Year==year)]
    
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(12, 10))
    plt.title(f'Propensity vs Cluster Size {year}', fontsize=20)
    sns.set_color_codes("pastel")
    sns.barplot(x='All_nests',
        y='ClusterID', data=df,
        label='Nest size', color='b')
    sns.set_color_codes("muted")    
    sns.barplot(x='Total_propensity', 
        y='ClusterID', data=df,
        label='Propensity', color='b')

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(-5, 25), ylabel='ClusterID',
        xlabel='')
    for i in ax.containers:
        ax.bar_label(i,)
    #ax.set_xticklabels(['5','0','5','10','15','20','25'])
    sns.despine(left=True, bottom=True)
    ax.get_figure().savefig(f'resources/visualisations/cluster_propensity{year}.png')

    df = cluster_data_by_site[(cluster_data_by_site['Year']==year)]
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(12, 10))
    plt.title(f'Propensity vs Site size {year}', fontsize=20)
    sns.set_color_codes("pastel")
    sns.barplot(x='All_nests',
        y='Site', data=df,
        label='Nest size', color='b')
    sns.set_color_codes("muted")    
    sns.barplot(x='Total_propensity', 
        y='Site', data=df,
        label='Propensity', color='b')
    
    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(-5, 25), ylabel='Site',
        xlabel='')
    #ax.set_xticklabels(['5','0','5','10','15','20','25'])
    for bars in ax.containers:
        ax.bar_label(bars,)
    sns.despine(left=True, bottom=True)
    ax.get_figure().savefig(f'resources/visualisations/cluster_site_propensity{year}.png')