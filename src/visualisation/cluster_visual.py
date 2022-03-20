"""
  This python script is mainly used to create different kind of visualisations of the data.
  Both maps and plots are created.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def generate_cluster_info(cluster: int, year: int, general_data: pd.DataFrame, 
    nests: pd.DataFrame, cluster_size_data: pd.DataFrame, dist:int):
    _, _, df_location_data  = fetch_data()
    all_nests = len([nestID for nestID in nests.index.values])
    nests_with_data = len([nestID for nestID in nests.index.values if nestID in general_data.index])
    sites = list(set([df_location_data.loc[nestID]['Site'].strip() for nestID in nests.index.values]))
    total_propensity = sum([general_data.loc[nestID]['Propensity']  if nestID in general_data.index else 0.2 for nestID in nests.index.values])
    row = pd.DataFrame({'ClusterID': [f'ClusterID {cluster}'], 'Year': [year], 'Dist': [dist], 'Site': [sites], 'Nests_with_data': [nests_with_data],
        'All_nests': [all_nests], 'Total_propensity': [total_propensity]})
    return cluster_size_data.append(row, ignore_index=True)

# Create site size vs propensity data
def create_site_propensity(years: list) -> pd.DataFrame:
    cluster_data_by_site = pd.DataFrame(columns=['Year', 'Site', 'Total_angrybirds', 'All_nests'])
    df_general_data, _, df_location_data  = fetch_data()
    for year in years:
        for site in df_location_data['Site'].str.strip().unique():
            total_propensity = sum([np.sum(df_general_data[(df_general_data['Site']==site2)&(df_general_data['Year']==year)]['Propensity']) if site == site2 else 0 for site2 in df_general_data['Site'].str.strip().unique()])
            size_nest_data = [df_general_data[(df_general_data['Site']==site2)&(df_general_data['Year']==year)] for site2 in df_general_data['Site'].str.strip().unique() if site == site2]
            if not size_nest_data:
                size_nest_data = 0
            else:
                size_nest_data = len(size_nest_data[0])
            size_all = len([nestID for nestID in df_location_data[(df_location_data['Site'] == site)&(df_location_data['Year'] == year)].index.values])
            row = pd.DataFrame({'Site': [site], 'Year': [year], 'Total_propensity': [total_propensity], 'All_nests': [size_all], 'Nests_with_data': [size_nest_data]})
            cluster_data_by_site = cluster_data_by_site.append(row, ignore_index=True)
    return  cluster_data_by_site

# Create cluster size vs propensity data
def create_cluster_propensity(years: list, cluster_dist: list) -> pd.DataFrame:
    cluster_size_data  = pd.DataFrame(columns=['ClusterID', 'Year', 'Dist', 'Site',
        'Nests_with_data','All_nests','Total_angrybirds'])
    cluster_size_data = cluster_size_data.astype({'ClusterID': str})
    df_general_data, df_clusters, _ = fetch_data()
    for dist in cluster_dist:
        for year in years:
            nests_current_year = df_clusters[df_clusters['Year'] == year]
            for i in range(min(df_clusters[f'ClusterID_{dist}']),max(df_clusters[f'ClusterID_{dist}'])+1):
                nests = nests_current_year[nests_current_year[f'ClusterID_{dist}'] == i]
                if len(nests) > 0:
                    cluster_size_data = generate_cluster_info(i, year, df_general_data, nests, cluster_size_data, dist)
    return cluster_size_data

def create_plots_cluster_propensity():
    _, df_clusters, _ = fetch_data()
    YEARS = df_clusters['Year'].unique()
    cluster_dist= [15,30,50,100, 200]
    cluster_size_data = create_cluster_propensity(YEARS, cluster_dist)
    site_size_data = create_site_propensity(YEARS)
    for year in YEARS:
        df = cluster_size_data[(cluster_size_data.Dist==100) & (cluster_size_data.Year==year)]
        
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
        ax.set(xlim=(0, 25), ylabel='ClusterID',
            xlabel='')
        for i in ax.containers:
            ax.bar_label(i,)
        sns.despine(left=True, bottom=True)
        ax.get_figure().savefig(f'resources/visualisations/plots/cluster_propensity{year}.png')

        df = site_size_data[(site_size_data['Year']==year)]
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
        ax.set(xlim=(0, 25), ylabel='Site',
            xlabel='')
        for bars in ax.containers:
            ax.bar_label(bars,)
        sns.despine(left=True, bottom=True)
        ax.get_figure().savefig(f'resources/visualisations/plots/cluster_site_propensity{year}.png')

def plot_mobbing_vs_cluster_size():
    _, df_clusters, _ = fetch_data()
    YEARS = df_clusters['Year'].unique()
    cluster_dist= [15,30,50,100, 200]
    cluster_size_data = create_cluster_propensity(YEARS, cluster_dist)
    for dist in cluster_dist:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10,15))
        plt.subplot(2,1,1)
        plt.title(f'Propensity Compared to Cluster Size (distance {dist} m)', fontsize=20)
        sns.stripplot(x=cluster_size_data[(cluster_size_data.Dist==dist) ]['All_nests'],
            y=cluster_size_data[(cluster_size_data.Dist==dist)]['Total_propensity'],
            jitter=0.3)
        plt.ylabel('Amount of Attacks (no data nests counts for 0.2))',fontsize=15)
        plt.savefig(f'resources/visualisations/plots/mobbing_vs_cluster_size{dist}.png')

plot_mobbing_vs_cluster_size()
create_plots_cluster_propensity()