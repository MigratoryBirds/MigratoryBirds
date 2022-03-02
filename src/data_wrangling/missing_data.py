import pandas as pd
import numpy as np

missing_cuckoo_dist = {
    '19EK16': 13.0,
    '19KR12': 44.0,
    '19KR13': 33.0,
    '19KR17': 50.0,
    '19KR18': 50.0,
    '19KR19': 50.0,
    '19EK28': 14.0,
    '19KR23': 90.0,
    '19KR26': 90.0,
    '19JJ6': 23.0,
    '19KR24': 27.75,
    '19KR25': 27.75,
    '19KR29': 41.0,
    '19AT11': 41.0,
    '19AT12': 41.0,
    '19AT13': 41.0,
    '19AT14': 41.0,
    '19KR30': 39.0,
    '19KR31': 39.0,
    '19KR34': 39.0,
    '19KR32': 15.0,
    '21DT3': 21.0
}

df = pd.read_csv('resources/generated_data/joined_dataset.csv')
df.Cuckoo_perch = [
    missing_cuckoo_dist[df.NestID[i]] if np.isnan(df.Cuckoo_perch[i])
    else df.Cuckoo_perch[i]
    for i in range(df.shape[0])
]
df.to_csv('resources/generated_data/joined_dataset_filled.csv', index=False)
