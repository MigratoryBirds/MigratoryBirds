import pandas as pd

dataframe_mobbing = pd.read_csv("resources/original_data/FinlandMobbingDatafile.csv")
dataframe_location = pd.read_csv("resources/original_data/FinlandNestDatafile.csv")

dataframe_mobbing = dataframe_mobbing.drop(columns=['Site', 'Year'])

data = pd.merge(dataframe_mobbing, dataframe_location, left_on='NestID', right_on='NestID')
data.to_csv("resources/generated_data/joined_dataset.csv", index=False)
