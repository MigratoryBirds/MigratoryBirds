import pandas as pd
import numpy as np

years19_20 = pd.read_csv("resources/original_data/FinlandNestDatafile.csv")
year21 = pd.read_csv("resources/original_data/Finland_nestdata2021_mod.csv")

sites21 = year21['Site'].unique()

sites20 = years19_20[years19_20['Year'] == 2020]['Site'].unique()
sites19 = years19_20[years19_20['Year'] == 2019]['Site'].unique()

sites21 = list(set(sorted([site.strip() for site in sites21])))
sites20 = list(set(sorted([site.strip() for site in sites20])))
sites19 = list(set(sorted([site.strip() for site in sites19])))

s21 = []
s20 = []
s19 = []
for site in sites21:
  if site not in sites20 and site not in sites19:
    s21.append(site)
for site in sites20:
  if site not in sites21 and site not in sites19:
    s20.append(site)
for site in sites19:
  if site not in sites21 and site not in sites20:
    s19.append(site)

print('Without data')
print(f"2019: {s19}")
print(f"2020: {s20}")
print(f"2021: {s21}")

years19_20 = pd.read_csv("resources/original_data/FinlandMobbingDatafile.csv")
year21 = pd.read_csv("resources/original_data/Finland_ExperimentData2021_mod.csv")

sites21 = year21['Site'].unique()

sites20 = years19_20[years19_20['Year'] == 2020]['Site'].unique()
sites19 = years19_20[years19_20['Year'] == 2019]['Site'].unique()

sites21 = list(set(sorted([site.strip() for site in sites21])))
sites20 = list(set(sorted([site.strip() for site in sites20])))
sites19 = list(set(sorted([site.strip() for site in sites19])))

s21 = []
s20 = []
s19 = []
for site in sites21:
  if site not in sites20 and site not in sites19:
    s21.append(site)
for site in sites20:
  if site not in sites21 and site not in sites19:
    s20.append(site)
for site in sites19:
  if site not in sites21 and site not in sites20:
    s19.append(site)

print('With data')
print(f"2019: {s19}")
print(f"2020: {s20}")
print(f"2021: {s21}")