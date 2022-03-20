"""
Testing the feature extraction from the maps programmatically
"""

import pandas as pd
import numpy as np
import requests
import json

df_general_data = pd.read_csv(
  '../../resources/generated_data/joined_dataset.csv', index_col='NestID'
)
df_clusters = pd.read_csv(
  '../../resources/generated_data/clusters.csv', index_col='NestID'
)
df_location_data = pd.read_csv( 
  '../../resources/original_data/FinlandNestDatafile.csv', index_col='NestID'
)
df_location_data21 = pd.read_csv( 
  '../../resources/original_data/Finland_nestdata2021_mod.csv', index_col='NestID'
)
df_location_data21['Year'] = 2021
df_location_data = pd.concat([df_location_data, df_location_data21])

overpass_url = "http://overpass-api.de/api/interpreter"
with open('my_data.txt', 'a') as data_file:
  index_reached=False
  for idx, row in df_location_data.iterrows():
    if index_reached:
      overpass_query= f"""
      [out:json];
      (
      way({row['lat']-0.0008},{row['long']-0.0015},{row['lat']+0.0008},{row['long']+0.0015});
      );
      out;
      """
      response = requests.get(overpass_url,
        params={'data': overpass_query}
      )
      data_res = response.json()
      data = {idx: data_res}
      data_file.write(json.dumps(data))
      data_file.write('\n')
    # Features are fetched only for a few nests at time due the limits for the queries done. 
    # Thus, after an error in orders to fetch new nests onwards from the nest we stopped on the previous iteration,
    # we need to check the ID from the previous nest and add it below.
    if "SomeId" == idx:
      index_reached = True