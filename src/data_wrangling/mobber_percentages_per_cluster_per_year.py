import numpy as np
import pandas as pd
import numpy.random as npr
from scipy.stats import binom
import matplotlib.pyplot as plt

clusters = pd.read_csv("resources\\generated_data\\clusters.csv")

clusters_2019 = clusters[clusters["Year"] == 2019]
clusters_2020 = clusters[clusters["Year"] == 2020]
clusters_2021 = clusters[clusters["Year"] == 2021]


data = pd.read_csv("resources\\generated_data\\generated_data.csv")
data_nest = pd.read_csv("resources\\original_data\\FinlandNestDatafile.csv")

data_2019 = data[data["Year"] == 2019]
data_nest_2019 = data_nest[data_nest["Year"] == 2019]

data_2020 = data[data["Year"] == 2020]
data_nest_2020 = data_nest[data_nest["Year"] == 2020]

data_2021 = pd.read_csv("resources\\original_data\\Finland_ExperimentData2021.csv",encoding = "ISO-8859-1")
data_2021.dropna(subset = ["NestID"], inplace=True)
data_nest_2021 = pd.read_csv("resources\\original_data\\Finland_nestdata2021.csv",encoding = "ISO-8859-1")
data_nest_2021.dropna(subset = ["NestID"], inplace=True)


def percentages(data, data_nest):
    aggressive = sum(data["Propensity"])
    shy = len(data) - aggressive
    no_data = len(data_nest) - len(data)

    whole = aggressive + shy + no_data

    if whole == 0:
        return (0,0,0,0,0,0)
    aggressive_percentage = aggressive / whole
    shy_percentage = shy / whole
    no_data_percentage = no_data / whole
    return(shy,aggressive,no_data,shy_percentage,aggressive_percentage,no_data_percentage)

shy_2019,agg_2019,nd_2019,p_shy_2019,p_agg_2019,p_nd_2019 = percentages(data_2019,data_nest_2019)
shy_2020,agg_2020,nd_2020,p_shy_2020,p_agg_2020,p_nd_2020 = percentages(data_2020,data_nest_2020)
shy_2021,agg_2021,nd_2021,p_shy_2021,p_agg_2021,p_nd_2021 = percentages(data_2021,data_nest_2021)


def percentages_in_clusters(clusterID,clusters,data,data_nest):
    
    N = max(clusters[clusterID].values) + 1
    dataframe = pd.DataFrame(columns=[clusterID, "shy","aggressive","no_data","total","percentage_shy","percentage_aggressive","percentage_no_data"])

    for i in range(N):
        new_cluster = clusters[clusters[clusterID] == i]
        new_data = new_cluster.merge(data, on="NestID")
        new_data_nest = new_cluster.merge(data_nest, on="NestID")
        
        shy, agg, nd,p_shy, p_agg, p_nd = percentages(new_data,new_data_nest)
        
        dataframe.loc[len(dataframe.index)] = [i, shy, agg, nd, shy+agg+nd ,p_shy, p_agg, p_nd]
        
    return dataframe




clusterIDs = ["ClusterID_15","ClusterID_30","ClusterID_50","ClusterID_100","ClusterID_200","ClusterID_300"]
dfs_2019 = [0] * 6

i = 0
for c in clusterIDs:
    
    df = percentages_in_clusters(c,clusters_2019,data_2019,data_nest_2019)
    df.to_csv(c + "_persentages.csv")
    df = df[df.total > 0]
    dfs_2019[i] = df
    i += 1



clusterIDs = ["ClusterID_15","ClusterID_30","ClusterID_50","ClusterID_100","ClusterID_200","ClusterID_300"]
dfs_2020 = [0] * 6

i = 0
for c in clusterIDs:
    
    df = percentages_in_clusters(c,clusters_2020,data_2020,data_nest_2020)
    df.to_csv(c + "_persentages.csv")
    df = df[df.total > 0]
    dfs_2020[i] = df
    i += 1

clusterIDs = ["ClusterID_15","ClusterID_30","ClusterID_50","ClusterID_100","ClusterID_200","ClusterID_300"]
dfs_2021 = [0] * 6

i = 0
for c in clusterIDs:
    
    df = percentages_in_clusters(c,clusters_2021,data_2021,data_nest_2021)
    df.to_csv(c + "_persentages.csv")
    df = df[df.total > 0]
    dfs_2021[i] = df
    i += 1

shy = shy_2019 + shy_2020 + shy_2021
agg = agg_2019 + agg_2020 + agg_2021
nd = nd_2019 + nd_2020 + nd_2021

ka_2019 = agg_2019 / (agg_2019 + shy_2019 + nd_2019)
ks_2019 = shy_2019 / (agg_2019 + shy_2019 + nd_2019)

ka_2020 = agg_2020 / (agg_2020 + shy_2020 + nd_2020)
ks_2020 = shy_2020 / (agg_2020 + shy_2020 + nd_2020)

ka_2021 = agg_2021 / (agg_2021 + shy_2021 + nd_2021)
ks_2021 = shy_2021 / (agg_2021 + shy_2021 + nd_2021)





def draw_aggressive(dfs_year,year,ka_year):

    fontsize = 15
    plt.rcParams.update({'font.size': fontsize})

    fig, ax = plt.subplots(12,figsize=(15,45))
    j = 0



    while j < 12:
        df = dfs_year[j//2]
        n = len(df)
        p_agg = [0] * n
        x = np.zeros(n)
        y = np.zeros(n)
        df = df.set_index([np.arange(n)])
        #print(df)

        for i in range(n):
        
            n_temp = df.at[i,"shy"] + df.at[i,"aggressive"] + df.at[i,"no_data"]
            
            x[i] = df.at[i,"percentage_aggressive"]
            n_agg = npr.binomial(n_temp, ka_year)
            y[i] = n_agg / n_temp
            p_agg[i] = binom.pmf(df.at[i,"aggressive"],n_temp,ka_year)
        
        colors = ['tomato']
        ax[j].hist(x,histtype='bar', color=colors, label=["aggressive"])
        ax[j].legend(prop={'size': fontsize})
        ax[j].set_title('bars with legend')
    
        ax[j].set_title(clusterIDs[j//2])

        ax[j].set_xlabel('percentage')
        ax[j].set_ylabel('frequency')
    
        ax[j+1].hist(y,histtype='bar', color=colors, label=["aggressive"])
        ax[j+1].legend(prop={'size': fontsize})
        ax[j+1].set_title('bars with legend')
    
        ax[j+1].set_title("Expectations for " + clusterIDs[j//2])
    
        ax[j+1].set_xlabel('percentage')
        ax[j+1].set_ylabel('frequency')

        j = j + 2
    
    title = 'Aggressive percentages for ' + str(year)
    fig.suptitle(title, fontsize=30)
    fig.tight_layout(pad=2.5)
    fig.patch.set_alpha(1)
    name = "resources\\visualisations\\plots\\percentage_aggressive_" + str(year) + ".png"
    fig.savefig(name, transparent=False)


draw_aggressive(dfs_2019,"2019",ka_2019)
draw_aggressive(dfs_2020,"2020",ka_2020)
draw_aggressive(dfs_2021,"2021",ka_2021)




def draw_shy(dfs_year,year,ks_year):

    fontsize = 15
    plt.rcParams.update({'font.size': fontsize})

    fig, ax = plt.subplots(12,figsize=(15,45))
    j = 0

    while j < 12:
        df = dfs_2019[j//2]
        n = len(df)

        p_shy = [0] * n
        x = np.zeros(n)
        y = np.zeros(n)
        df = df.set_index([np.arange(n)])

        for i in range(n):
        
            x[i] = df.at[i,"percentage_shy"]
        
            n_temp = df.at[i,"shy"] + df.at[i,"aggressive"] + df.at[i,"no_data"]
            n_shy = npr.binomial(n_temp, ks_year)
            y[i] = n_shy / n_temp
            p_shy[i] = binom.pmf(df.at[i,"shy"],n_temp,ks_year)
        
        colors = ['cornflowerblue']
        ax[j].hist(x,histtype='bar', color=colors, label=["shy"])
        ax[j].legend(prop={'size': fontsize})
        ax[j].set_title('bars with legend')
    
        ax[j].set_title(clusterIDs[j//2])

        ax[j].set_xlabel('percentage')
        ax[j].set_ylabel('frequency')
    
        ax[j+1].hist(y,histtype='bar', color=colors, label=["shy"])
        ax[j+1].legend(prop={'size': fontsize})
        ax[j+1].set_title('bars with legend')
    
        ax[j+1].set_title("Expectations for " + clusterIDs[j//2])
    
        ax[j+1].set_xlabel('percentage')
        ax[j+1].set_ylabel('frequency')

        j = j + 2
    
    title = 'Shy percentages for ' + str(year)
    fig.suptitle(title, fontsize=30)
    fig.tight_layout(pad=2.5)
    fig.patch.set_alpha(1)
    name = "resources\\visualisations\\plots\\percentage_shy_" + str(year) + ".png"
    fig.savefig(name, transparent=False)
    
    
draw_shy(dfs_2019,"2019",ks_2019)
draw_shy(dfs_2020,"2020",ks_2020)
draw_shy(dfs_2021,"2021",ks_2021)