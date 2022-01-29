import os
import pandas as pd

def import_wine_data():
  '''
  function to import wine review raw data into pandas dataframe
  '''
  folder_url = r'raw_data/wine_reviews/'
  wine_raw_data = pd.DataFrame()
  for file in os.listdir(folder_url):
    file_url = folder_url + '/' + file
    data_to_append = pd.read_csv(file_url, encoding='latin1', low_memory=False)
    wine_raw_data = pd.concat([wine_raw_data, data_to_append], axis=0, ignore_index=True)

  wine_raw_data.drop_duplicates(subset=['Name'], inplace=True)
  for geo in ['Subregion', 'Region', 'Province', 'Country']:
    wine_raw_data[geo] = wine_raw_data[geo].apply(lambda g: str(g).strip())
  
  return wine_raw_data


def import_food_data():
  '''
  function to import food review data into pandas dataframe
  '''
  return pd.read_csv('raw_data/amazon_food_reviews/Reviews.csv', low_memory=False)


def import_descriptor_mapping():
  return pd.read_csv('references/descriptor_mapping.csv', encoding='latin1').set_index('raw descriptor')