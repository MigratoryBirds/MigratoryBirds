""" This file contains the functions needed to perform the join count analysis. 
"""
import pandas as pd
import numpy as np
import geopy.distance


df = pd.read_csv('resources/generated_data/joined_dataset.csv', index_col = False)

#-----------------------------------------------------------------------------------------------------------------

def create_distance_matrix(df):
    """ Creates a matrix where the value at (row i, column j) is the distance between observed bird 'i' and observed bird 'j'"""

    #setting index to be observation number
    df.index = range(df.shape[0])

    obsv = df.shape[0] #observations  

    dist_mtx = np.zeros((obsv,obsv)) #matrix indicating which observations are neighbors with each other
    
    #filling in the neighbor matrices
    for index,row in df.iterrows():
        lat,long = row['lat'],row['long']
        for index2,row2 in df.iterrows():
            lat2,long2 = row2['lat'],row2['long']
            coords1 = (lat,long)
            coords2 = (lat2,long2)
            dist_mtx[index,index2] = geopy.distance.distance(coords1,coords2).km*1000
    
    return dist_mtx

#----------------------------------------------------------------------------------------------------------------------------------

def create_neighbor_matrix(df,dis_threshold):
    """ Creates a matrix where values are either 1 or 0.
        A value of 1 at (row i, column j) indicates that observed bird 'i' and observed bird 'j' are less than or equal to 'dis_threshold' meters apart"""

    obsv = df.shape[0] #observations  

    neighbor_mtx = np.zeros((obsv,obsv)) #matrix indicating which observations are neighbors with each other
    
    dm = create_distance_matrix(df)

    #filling in the neighbor matrices
    for i in range(obsv):
        for j in range(obsv):
            if i==j:
                continue
            if (dm[i,j] <= dis_threshold):
                neighbor_mtx[i,j] = 1
    
    return neighbor_mtx

#----------------------------------------------------------------------------------------------------------------------------------

def join_counts(df,dis_threshold):
    """ dis_threshold is in m
        takes in array of distances b/w observations and outputs a dictionary containing the number of join counts of each type"""

    obsv = df.shape[0] #observations  

    neighbor_mtx = create_neighbor_matrix(df,dis_threshold) #matrix indicating which observations are neighbors with each other
    
    # aggressive-aggressive neighbors, shy-shy, aggressive-shy
    AA,SS,AS = 0,0,0 

    values = df['Propensity'].to_numpy().reshape(obsv,1)

    #filling in the join counts
    for i in range(obsv):
        for j in range(obsv):
            if neighbor_mtx[i,j] == 1:
                if values[i] == 1 and values[j] == 1:
                    AA += 1
                elif values[i] == 0 and values[j] == 0:
                    SS += 1
                else:
                    AS += 1

    #halving because of matrix symmetry
    AA = AA * 0.5
    SS = SS * 0.5
    AS = AS * 0.5

    return {'AA':AA,'SS':SS,'AS':AS}    

#----------------------------------------------------------------------------------------------------------------------------------

def random_join_counts(df, dis_threshold, repetitions = 1) -> dict:
    """ Generates the random join counts by shuffling all the birds among each other but keeping the location of the nests the same. 
        Can specify the number of simulations to be done. Outputs a dictionary with all the different counts of the different simulations"""

    obsv = df.shape[0] #observations 

    neighbor_mtx = create_neighbor_matrix(df, dis_threshold)

    one_counts = int(neighbor_mtx.sum() / 2)

    col_count = int(obsv * (obsv - 1) / 2)

    vector_zeros = np.zeros((1,col_count - one_counts))

    vector_ones = np.ones((1,one_counts))

    random_order = np.concatenate((vector_ones,vector_zeros),axis=None)

    values = df['Propensity'].to_numpy().reshape(obsv,1)

    rand_neighbor_mtx = np.zeros((obsv,obsv))

    reps = 0

    results = {'AA':[],'SS':[],'AS':[]}

    while(reps < repetitions):

        np.random.shuffle(random_order)

        ind = 0

        for i in range(obsv):
            for j in range(obsv):
                if i==j:
                    rand_neighbor_mtx[i,j] = 0
                elif i > j:
                    rand_neighbor_mtx[i,j] = rand_neighbor_mtx[j,i]
                else:
                    rand_neighbor_mtx[i,j] = random_order[ind]
                    ind += 1
        
        AA,SS,AS = 0,0,0

        for i in range(obsv):
            for j in range(obsv):
                if rand_neighbor_mtx[i,j] == 1:
                    if values[i] == 1 and values[j] == 1:
                        AA += 1
                    elif values[i] == 0 and values[j] == 0:
                        SS += 1
                    else:
                        AS += 1

        #halving because of matrix symmetry
        AA = AA * 0.5
        SS = SS * 0.5
        AS = AS * 0.5

        results['AA'].append(AA)
        results['SS'].append(SS)
        results['AS'].append(AS)

        reps += 1

    return results

    #----------------------------------------------------------------------------------------------------------------------------------