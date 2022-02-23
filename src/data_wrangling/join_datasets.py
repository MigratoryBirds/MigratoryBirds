import pandas as pd

# dataframe_mobbing_1 = pd.read_csv("resources/original_data/FinlandMobbingDatafile.csv")
# dataframe_mobbing_2 = pd.read_csv("resources/original_data/Finland_ExperimentData2021_mod.csv")
# dataframe_location_1 = pd.read_csv("resources/original_data/FinlandNestDatafile.csv")
# dataframe_location_2 = pd.read_csv("resources/original_data/Finland_nestdata2021_mod.csv")
# dataframe_location_2['Year'] = 2021

# dataframe_mobbing_2.drop(['lat', 'long'], axis=1, inplace=True)

# columns_to_switch =
# dataframe_location_2[] = data_frame_mobbing_2[]
# dataframe_location_2.drop(columns_to_switch, axis=1, inplace=True)

# dataframe_mobbing = pd.concat([dataframe_mobbing_1, dataframe_mobbing_2])
# dataframe_location = pd.concat([dataframe_location_1, dataframe_location_2])

# dataframe_mobbing = dataframe_mobbing.drop(columns=['Site', 'Year'])

# print(dataframe_mobbing.columns)
# print(dataframe_location.columns)

# data = pd.merge(dataframe_mobbing, dataframe_location, left_on='NestID', right_on='NestID')
# data.to_csv("resources/generated_data/joined_dataset.csv", index=False)


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
print(data_1)
data_2 = pd.merge(
    dataframe_mobbing_2,
    dataframe_location_2,
    left_on='NestID',
    right_on='NestID'
)
print(data_2)

data = pd.concat([data_1, data_2])
data.to_csv("resources/generated_data/joined_dataset.csv", index=False)
