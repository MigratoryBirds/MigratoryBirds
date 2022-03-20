"""
Make csv file where the differences between the years per site is fast to see at once.
In progress...
"""

from utils.data_utils import fetch_data
import pandas as pd


df_general, df_clusters, df_location = fetch_data()

sites = df_location.Site.str.strip().unique()
df = pd.DataFrame(index=sites)
for year in df_location.Year.unique():
  sizes = []
  propensities = []
  shyness = []
  no_data = []
  max_time_between_trial_nest_found = ""
  mean_time_between_trial_nest_found = ""
  min_time_between_trial_nest_found =""
  for site in sites:
    all = len(df_location[(df_location.Year==year) & (df_location.Site.str.strip()==site)])
    sizes.append(all)
    shy = -1
    aggressive = -1
    if site in df_general[df_general.Year==year].Site.str.strip().unique():
      aggressive = len(df_general[(df_general.Year==year) & (df_general.Site==site) & (df_general.Propensity==1)])
      shy = len(df_general[(df_general.Year==year) & (df_general.Site==site) & (df_general.Propensity==0)])
    propensities.append(aggressive)
    shyness.append(shy)
    if shy < 0:
      shy = 0
    if aggressive < 0:
      aggressive = 0
    no_data.append(all-shy-aggressive)

  df[f'Size{year}'] = sizes
  df[f'Aggressive{year}'] = propensities
  df[f'Shy{year}'] = shyness
  df[f'No_data{year}'] = no_data


df = df.reindex(columns=sorted(df.columns, reverse=False))
df = df.sort_values(by=['Size2019'], ascending=False)
df.to_csv("resources/generated_data/site_differences.csv")