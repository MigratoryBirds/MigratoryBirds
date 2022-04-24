"""
This script produces six plots and an excel sheet with the amount of certain types of nests each year and their percentages. 
Two of the plots are presenting percentages of aggressive and shy birds in clusters 
with the no data nests classified randomly either as aggressive or shy with weighted likelihood or 50-50 likelihood.
One presents the percentages of all the nest types (aggressive, shy and no data) in the plot. 
Three remaining plots are control plots for these, showing the percentages in each cluster
when the nesting behavior is considered random. 
"""


import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt

clusters = pd.read_csv("resources\\generated_data\\clusters.csv")

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

    aggressive_percentage = aggressive / whole
    shy_percentage = shy / whole
    no_data_percentage = no_data / whole
    return(shy,aggressive,no_data,shy_percentage,aggressive_percentage,no_data_percentage)

shy_2019,agg_2019,nd_2019,p_shy_2019,p_agg_2019,p_nd_2019 = percentages(data_2019,data_nest_2019)
shy_2020,agg_2020,nd_2020,p_shy_2020,p_agg_2020,p_nd_2020 = percentages(data_2020,data_nest_2020)
shy_2021,agg_2021,nd_2021,p_shy_2021,p_agg_2021,p_nd_2021 = percentages(data_2021,data_nest_2021)

percentages_per_year = pd.DataFrame(columns=["Year","shy","aggressive","no_data","percentage_shy","percentage_aggressive","percentage_no_data"])
percentages_per_year.loc[len(percentages_per_year.index)] = ["2019",shy_2019,agg_2019,nd_2019, p_shy_2019, p_agg_2019,p_nd_2019]
percentages_per_year.loc[len(percentages_per_year.index)] = ["2020",shy_2020,agg_2020,nd_2020,p_shy_2020,p_agg_2020,p_nd_2020]
percentages_per_year.loc[len(percentages_per_year.index)] = ["2021", shy_2021,agg_2021,nd_2021,p_shy_2021,p_agg_2021,p_nd_2021]
percentages_per_year.to_csv("resources\\generated_data\\" + "persentages_per_year.csv")

def percentages_in_clusters(clusterID,clusters,data,data_nest):
    
    N = max(clusters[clusterID].values) + 1
    dataframe = pd.DataFrame(columns=[clusterID, "shy","aggressive","no_data","percentage_shy","percentage_aggressive","percentage_no_data"])

    for i in range(N):
        
        new_cluster = clusters[clusters[clusterID] == i]
        new_data = new_cluster.merge(data, on="NestID")
        new_data_nest = new_cluster.merge(data_nest, on="NestID")
        
        shy, agg, nd,p_shy, p_agg, p_nd = percentages(new_data,new_data_nest)
        
        dataframe.loc[len(dataframe.index)] = [i, shy, agg, nd,p_shy, p_agg, p_nd]
        
    return dataframe

data = data.append(data_2021, ignore_index=True)
data_nest = data_nest.append(data_nest_2021, ignore_index=True)


clusterIDs = ["ClusterID_15","ClusterID_30","ClusterID_50","ClusterID_100","ClusterID_200","ClusterID_300"]

dfs = [0]*len(clusterIDs)
i = 0

for c in clusterIDs:
    
    df = percentages_in_clusters(c,clusters,data,data_nest)
    df.to_csv("resources\\generated_data\\" + c + "_persentages.csv")
    dfs[i] = df
    i += 1


def draw_plots3(dfs,name):

    fig, ax = plt.subplots(6,figsize=(15,15))

    for j in range(len(dfs)):
        df = dfs[j]
        n = len(df)
        x = np.zeros((n,3))

        for i in range(n):
            x[i,0] = df.at[i,"percentage_shy"]
            x[i,1] = df.at[i,"percentage_aggressive"]
            x[i,2] = df.at[i,"percentage_no_data"]
        
        colors = ['cornflowerblue', 'tomato', 'grey']
        ax[j].hist(x,histtype='bar', color=colors, label=["shy","aggressive","no data"])
        ax[j].legend(prop={'size': 10})
        ax[j].set_title('bars with legend')
    
        ax[j].set_title(clusterIDs[j])
    
        ax[j].set_xlabel('percentage')
        ax[j].set_ylabel('frequency')

    print()
    fig.tight_layout(pad=2.0)
    fig.patch.set_alpha(1)
    fig.savefig("resources\\visualisations\\plots\\" + name + ".png", transparent=False)


def draw_plots2(dfs,multiplier_shy,multiplier_agg,name):

    fig, ax = plt.subplots(6,figsize=(15,15))

    for j in range(len(dfs)):
        df = dfs[j]
        n = len(df)
        x = np.zeros((n,2))

        for i in range(n):
            x[i,0] = df[df[clusterIDs[j]] == i].at[i,"percentage_shy"] + df[df[clusterIDs[j]] == i].at[i,"percentage_no_data"] * multiplier_shy
            x[i,1] = df[df[clusterIDs[j]] == i].at[i,"percentage_aggressive"] + df[df[clusterIDs[j]] == i].at[i,"percentage_no_data"] * multiplier_agg
    
        colors = ['cornflowerblue', 'tomato']
        ax[j].hist(x,histtype='bar', color=colors, label=["shy","aggressive"])
        ax[j].legend(prop={'size': 10})
        ax[j].set_title('bars with legend')
    
        ax[j].set_title(clusterIDs[j])
    
        ax[j].set_xlabel('percentage')
        ax[j].set_ylabel('frequency')

    print()
    fig.tight_layout(pad=2.0)
    fig.patch.set_alpha(1)
    fig.savefig("resources\\visualisations\\plots\\" + name + ".png", transparent=False)


shy = shy_2019 + shy_2020 + shy_2021
agg = agg_2019 + agg_2020 + agg_2021
nd = nd_2019 + nd_2020 + nd_2021
ka = agg / (agg + shy)
ks = shy / (agg + shy)

draw_plots3(dfs,"percentage_with_no_data")
draw_plots2(dfs,ks,ka,"percentage_with_weighted_split")
draw_plots2(dfs,0.5,0.5,"percentage_with_50_50_split")

    
def draw_plots2_control(dfs,name,ka,ks):

    fig, ax = plt.subplots(6,figsize=(15,15))

    for j in range(len(dfs)):
        df = dfs[j]
        n = len(df)
        x = np.zeros((n,2))

        for i in range(n):
            
            n = df.at[i,"shy"] + df.at[i,"aggressive"] + df.at[i,"no_data"]
            n_agg = npr.binomial(n, ka)
            n_shy = npr.binomial(n, ks)
            n_nd = max(0,n-n_agg -n_shy )
            n_new = n_agg + n_shy + n_nd
            
            if(n_new == 0): n_new = 1
                
            x[i,0] = n_shy / n_new  + n_nd/n_new *ks
            x[i,1] = n_agg / n_new  + n_nd/n_new *ka
            #print(x[i,0],x[i,1])
        
        colors = ['cornflowerblue', 'tomato']
        ax[j].hist(x,histtype='bar', color=colors, label=["shy","aggressive"])
        ax[j].legend(prop={'size': 10})
        ax[j].set_title('bars with legend')
    
        ax[j].set_title(clusterIDs[j])
    
        ax[j].set_xlabel('percentage')
        ax[j].set_ylabel('frequency')

    print()
    fig.tight_layout(pad=2.0)
    fig.patch.set_alpha(1)
    fig.savefig("resources\\visualisations\\plots\\" + name + ".png", transparent=False)



def draw_plots3_control(dfs,name,ka,ks,nd):

    fig, ax = plt.subplots(6,figsize=(15,15))

    for j in range(len(dfs)):
        df = dfs[j]
        n = len(df)
        x = np.zeros((n,3))

        for i in range(n):
            n = df.at[i,"shy"] + df.at[i,"aggressive"] + df.at[i,"no_data"]
            n_agg = npr.binomial(n, ka)
            n_shy = npr.binomial(n, ks)
            n_nd = npr.binomial(n, nd)
            n_new = n_agg + n_shy + n_nd
            if(n_new == 0): n_new = 1
            x[i,0] = n_shy / n_new 
            x[i,1] = n_agg / n_new 
            x[i,2] = n_nd / n_new 
            #print(x[i,0],x[i,1],x[i,2])
        
        colors = ['cornflowerblue', 'tomato', 'grey']
        ax[j].hist(x,histtype='bar', color=colors, label=["shy","aggressive","no data"])
        ax[j].legend(prop={'size': 10})
        ax[j].set_title('bars with legend')
    
        ax[j].set_title(clusterIDs[j])
    
        ax[j].set_xlabel('percentage')
        ax[j].set_ylabel('frequency')

    print()
    fig.tight_layout(pad=2.0)
    fig.patch.set_alpha(1)
    fig.savefig("resources\\visualisations\\plots\\" + name + ".png", transparent=False)


draw_plots2_control(dfs,"percentage_with_weighted_split_control",ka,ks)
draw_plots2_control(dfs,"percentage_with_50_50_split_control",0.5,0.5)

    
shy = shy_2019 + shy_2020 + shy_2021
agg = agg_2019 + agg_2020 + agg_2021
nd = nd_2019 + nd_2020 + nd_2021
ka = agg / (agg + shy + nd)
ks = shy / (agg + shy + nd)
nd = nd / (agg + shy + nd)

draw_plots3_control(dfs,"percentage_with_no_data_control",ka,ks,nd)

