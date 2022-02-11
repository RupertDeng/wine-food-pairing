import ast
import os
import re
import pandas as pd
import numpy as np
from gensim.models.phrases import Phraser
from gensim.models import Word2Vec

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
  return pd.read_csv('references/descriptor_mapping_tastes.csv', encoding='latin1').set_index('raw descriptor')


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
  """
  return a dataframe with different variety/geo columns normalized to a single column of geo data
  """
  return pd.read_csv('references/varieties_all_geos_normalized.csv', index_col=0)


def import_wine_phraser():
  """
  import trained wine trigram model from step1_train_word_embedding
  """
  return Phraser.load('trained_models/wine_trigram_model.pkl')


def import_food_phraser():
  """
  import trained food trigram model from step1_train_word_embedding
  """
  return Phraser.load('trained_models/food_trigram_model.pkl')


def import_word2vec_model():
  """
  import trained word2vec model from step1_train_word_embedding
  """
  return Word2Vec.load('trained_models/wine_food_word2vec_model.pkl')


def import_descriptorized_wine_data():
  """
  import cleaned-up and descriptorized wine dataframe from step2_prepare_wine_data_set
  """
  return pd.read_csv('processed_data/descriptorized_wine_df.csv', keep_default_na=False)


def import_list_of_foods():
  """
  return a comprehensive list of food as dataframe
  """
  df = pd.read_csv('references/list_of_foods.csv')
  return list(df['Food'])


def nparray_str_to_list(arr):
  vector = re.sub('\s+', ',', arr).replace('[,', '[')
  return np.array(ast.literal_eval(vector))


def import_wine_variety_vector_info():
  wine_vector_df = pd.read_csv('processed_data/wine_variety_vector.csv', index_col=0)
  wine_vector_df['aroma vector'] = wine_vector_df['aroma vector'].map(nparray_str_to_list)
  return wine_vector_df


def import_wine_variety_descriptor_info():
  return pd.read_csv('processed_data/wine_variety_aroma_descriptor.csv', index_col=0)


def import_food_nonaroma_info():
  food_nonaroma_df = pd.read_csv('processed_data/food_nonaroma_df.csv', index_col=0)
  food_nonaroma_df['average_vec'] = food_nonaroma_df['average_vec'].map(nparray_str_to_list)
  return food_nonaroma_df

