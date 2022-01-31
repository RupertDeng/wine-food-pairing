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

  wine_raw_data.drop(wine_raw_data.columns[-1], axis=1, inplace=True)
  
  return wine_raw_data


def import_food_data():
  '''
  function to import food review data into pandas dataframe
  '''
  return pd.read_csv('raw_data/amazon_food_reviews/Reviews.csv', low_memory=False)


def import_descriptor_mapping():
  return pd.read_csv('references/descriptor_mapping.csv', encoding='latin1').set_index('raw descriptor')


def import_variety_mapping():
  variety_mapping = {'Shiraz': 'Syrah', 'Pinot Gris': 'Pinot Grigio', 'Pinot Grigio/Gris': 'Pinot Grigio',
  'Garnacha, Grenache': 'Grenache', 'Garnacha': 'Grenache', 'CarmenÃ¨re': 'Carmenere',
  'GrÃ¼ner Veltliner': 'Gruner Veltliner', 'TorrontÃ©s': 'Torrontes', 
  'RhÃ´ne-style Red Blend': 'Rhone-style Red Blend', 'AlbariÃ±o': 'Albarino',
  'GewÃ¼rztraminer': 'Gewurztraminer', 'RhÃ´ne-style White Blend': 'Rhone-style White Blend',
  'SpÃƒÂ¤tburgunder, Pinot Noir': 'Pinot Noir', 'Sauvignon, Sauvignon Blanc': 'Sauvignon Blanc',
  'Pinot Nero, Pinot Noir': 'Pinot Noir', 'Malbec-Merlot, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend',
  'Meritage, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend', 'Garnacha, Grenache': 'Grenache',
  'FumÃ© Blanc': 'Sauvignon Blanc', 'Cabernet Sauvignon-Cabernet Franc, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend',
  'Cabernet Merlot, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend', 'Cabernet Sauvignon-Merlot, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend',
  'Cabernet Blend, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend', 'Malbec-Cabernet Sauvignon, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend',
  'Merlot-Cabernet Franc, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend', 'Merlot-Cabernet Sauvignon, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend',
  'Cabernet Franc-Merlot, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend', 'Merlot-Malbec, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend',
  'Cabernet, Bordeaux-style Red Blend': 'Bordeaux-style Red Blend', 'Primitivo, Zinfandel': 'Zinfandel',
  'AragonÃªs, Tempranillo': 'Aragonez, Tempranillo'}
  return variety_mapping


def import_normalized_geo_data():
  return pd.read_csv('references/varieties_all_geos_normalized.csv', index_col=0)


def import_taste_descriptor_mapping(core_tastes):
  descriptors = pd.read_csv('references/descriptor_mapping_tastes.csv', encoding='latin1').set_index('raw descriptor')
  mapping = dict()
  for taste in core_tastes:
    if taste == 'aroma':
      mapping[taste] = descriptors.loc[descriptors['type'] == 'aroma']
    else:
      mapping[taste] = descriptors.loc[descriptors['primary taste'] == taste]
  return mapping