import numpy as np
import pandas as pd

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

clusterIDs = ["ClusterID_15","ClusterID_30","ClusterID_50","ClusterID_100","ClusterID_200","ClusterID_300"]

for c in clusterIDs:
    
    df = percentages_in_clusters(c,clusters,data,data_nest)
    df.to_csv("resources\\generated_data\\" + c + "_persentages.csv")

