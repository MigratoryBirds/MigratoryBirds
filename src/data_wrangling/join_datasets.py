import pandas as pd


dataframe_mobbing_1 = pd.read_csv(
    "resources/original_data/FinlandMobbingDatafile.csv"
).drop(
    'Year Site'.split(), axis=1, inplace=True
)
dataframe_mobbing_2 = pd.read_csv(
    "resources/original_data/Finland_ExperimentData2021_mod.csv"
).drop(
    'Year lat long Site Cuckoo_perch'.split(), axis=1, inplace=True
)
dataframe_location_1 = pd.read_csv(
    "resources/original_data/FinlandNestDatafile.csv"
)
dataframe_location_2 = pd.read_csv(
    "resources/original_data/Finland_nestdata2021_mod.csv"
)
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

data = pd.concat([data_1, data_2])
data.to_csv("resources/generated_data/joined_dataset.csv", index=False)
