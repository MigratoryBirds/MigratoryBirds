import pandas as pd

dataframe_mobbing = pd.read_csv("resources/original_data/FinlandMobbingDatafile.csv")
dataframe_location = pd.read_csv("resources/original_data/FinlandNestDatafile.csv")

data21 = pd.read_csv("resources/original_data/Finland_ExperimentData2021_mod.csv")

dataframe_mobbing = dataframe_mobbing.drop(columns=['Site', 'Year'])
data = pd.merge(dataframe_mobbing, dataframe_location, left_on='NestID', right_on='NestID')

data = pd.concat([data21,data])
data.to_csv("resources/generated_data/joined_dataset.csv", index=False)
