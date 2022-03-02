import pandas as pd

dataframe_mobbing_1 = pd.read_csv(
    "resources/original_data/FinlandMobbingDatafile.csv"
).drop(
    'Year Site'.split(), axis=1
)
dataframe_mobbing_2 = pd.read_csv(
    "resources/original_data/Finland_ExperimentData2021_mod.csv"
).drop(
    'Year lat long Site Cuckoo_perch'.split(), axis=1
)
dataframe_location_1 = pd.read_csv(
    "resources/original_data/FinlandNestDatafile.csv"
)
dataframe_location_1['Site'] = dataframe_location_1['Site'].str.strip()
dataframe_location_2 = pd.read_csv(
    "resources/original_data/Finland_nestdata2021_mod.csv"
)
dataframe_location_2['Site'] = dataframe_location_2['Site'].str.strip()
dataframe_location_2['Year'] = 2021
data_1 = pd.merge(
    dataframe_mobbing_1,
    dataframe_location_1,
    left_on='NestID',
    right_on='NestID'
)
data_2 = pd.merge(
    dataframe_mobbing_2,
    dataframe_location_2,
    left_on='NestID',
    right_on='NestID'
)
data = pd.concat([data_1, data_2]).set_index('NestID')

map_features = (
    pd.read_csv('resources/generated_data/map_features.csv')
    .drop(columns=['Site', 'X'])  # X is the last empty column
).set_index('NestID')

data = data.join(map_features)

data.to_csv("resources/generated_data/joined_dataset.csv")
