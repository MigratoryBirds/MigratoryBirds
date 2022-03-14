from ntpath import join
import pandas as pd
import numpy as np
import geopy.distance
from esda.join_counts import Join_Counts
import matplotlib.pyplot as plt

df = pd.read_csv('resources/generated_data/joined_dataset.csv', index_col = False)
df21 = pd.read_csv('resources/original_data/Finland_ExperimentData2021_mod.csv', index_col= False)

#seperating years
df19 = df[df.Year == 2019]
df20 = df[df.Year == 2020]

#setting index to be observation number
df19.index = range(df19.shape[0])
df20.index = range(df20.shape[0])
df21.index = range(df21.shape[0])

#-----------------------------------------------------------------------------------------------------------------

#dis_threshold is in km
def join_counts(df,dis_threshold) -> dict:

    #setting index to be observation number
    df.index = range(df.shape[0])

    obsv = df.shape[0] #observations  

    neighbor_mtx = np.zeros((obsv,obsv)) #matrix indicating which observations are neighbors with each other
    
    #filling in the neighbor matrices
    for index,row in df.iterrows():
        lat,long = row['lat'],row['long']
        for index2,row2 in df.iterrows():
            lat2,long2 = row2['lat'],row2['long']
            coords1 = (lat,long)
            coords2 = (lat2,long2)
            if (geopy.distance.distance(coords1,coords2).km <= dis_threshold):
                neighbor_mtx[index,index2] = 1
    
    y = df['Propensity'].to_numpy()

    y.reshape(obsv,1)

    #aggressive aggressive neighbors, shy shy, aggr shy
    AA,SS,AS = 0,0,0 

    #filling in the join counts
    for i in range(obsv):
        for j in range(obsv):
            if i==j:
                continue
            else:
                if neighbor_mtx[i,j] == 1:
                    if y[i] == 1 and y[j] == 1:
                        AA += 1
                    elif y[i] == 0 and y[j] == 0:
                        SS += 1
                    else:
                        AS += 1

    #halving because of matrix symmetry
    AA = AA * 0.5
    SS = SS * 0.5
    AS = AS * 0.5

    return {'AA':AA,'SS':SS,'AS':AS}

#----------------------------------------------------------------------------------------------------------------------------------

print(join_counts(df19,0.05))
print(join_counts(df20,0.05))
print(join_counts(df21,0.05))

#jc = Join_Counts(df19['Propensity'],neighbor_mtx_19)

#----------------------------------------------------------------------------------------------------------------------------------

#create matrix of distances between observations
def create_distance_matrix(df):

     #setting index to be observation number
    df19.index = range(df19.shape[0])

    obsv = df.shape[0] #observations  

    dist_mtx = np.zeros((obsv,obsv)) #matrix indicating which observations are neighbors with each other
    
    #filling in the neighbor matrices
    for index,row in df.iterrows():
        lat,long = row['lat'],row['long']
        for index2,row2 in df.iterrows():
            lat2,long2 = row2['lat'],row2['long']
            coords1 = (lat,long)
            coords2 = (lat2,long2)
            dist_mtx[index,index2] = geopy.distance.distance(coords1,coords2).km
    
    return dist_mtx

#----------------------------------------------------------------------------------------------------------------------------------

print(create_distance_matrix(df19))