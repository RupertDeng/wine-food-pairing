from data_importer import import_wine_data, import_variety_mapping, import_normalized_geo_data, import_taste_descriptor_mapping, import_wine_phraser
import pandas as pd
from step1_train_word_embedding import normalize_sentence
from dask_multiprocessing import dask_compute

def cleanup_variety(df, mapping):
  """normalize variety based on variety mapping"""
  df['Variety'] = df['Variety'].map(lambda v: mapping.get(v, v))
  return df


def replace_nan_with_none(value):
  """
  simple function to convert nan to 'none'
  """
  if str(value) in ['0', 'nan']:
    return 'none'
  else:
    return value


def cleanup_geo(df):
  """
  fill empty geo cells with 'none'
  """
  order_of_geographies = ['Subregion', 'Region', 'Province', 'Country']
  for geo in order_of_geographies:
    df[geo] = df[geo].map(replace_nan_with_none)
  df.loc[:, order_of_geographies].fillna('none', inplace=True)
  return df


def consolidate_variety_geo(df, mapping):
  """
  consolidate variety + geo with normalized mapping. The mapping has various variety + geo mapped to a uniform geo data.
  """
  merge_columns = ['Variety', 'Country', 'Province', 'Region', 'Subregion']
  df = pd.merge(left=df, right=mapping, left_on=merge_columns, right_on=merge_columns)
  df = df[['Name', 'Variety', 'geo_normalized', 'Description']]
  return df


def trim_by_variety_geo_frequency(df, freq):
  """
  function to remove less frequent variety + geo rows
  """
  trim_columns = ['Variety', 'geo_normalized']
  variety_geos = df.groupby(trim_columns).size().reset_index().rename(columns={0: 'count'})
  variety_geos = variety_geos.loc[variety_geos['count'] > freq]
  trimmed_df = pd.merge(left=df, right=variety_geos, left_on=trim_columns, right_on=trim_columns).drop(['count'], axis=1).dropna()
  return trimmed_df


def get_taste_descriptor(word, mapping_df):
  """
  function to get taste descriptor for certain word. If there is no available mapping, simply return ''.
  """
  try:
    return str(mapping_df.at[word, 'combined']).strip()
  except:
    return ''


def map_review_to_taste_descriptor(df, col, tokenizer, phraser, taste_mapping):
  """
  function to map each review in dataframe to a list of taste descriptors
  """
  def process_sent(sent):
    try:
      phrased_sent = phraser[tokenizer(sent)]
      mapped_sent = [get_taste_descriptor(word, taste_mapping) for word in phrased_sent]
      return ' '.join([word for word in mapped_sent if word])
    except:
      return ''
  
  df[col] = df[col].map(process_sent)
  return df


if __name__ == '__main__':

  # import raw wine data as dataframe
  wine_df = import_wine_data()
  
  # map the Variety values to normalized terms
  variety_mapping = import_variety_mapping()
  wine_df = cleanup_variety(wine_df, variety_mapping)

  # set empty geo cell to 'none'
  wine_df = cleanup_geo(wine_df)

  # map the geography columns to a single column of normalized geo and remove other columns not needed
  normalized_geo_df = import_normalized_geo_data()
  wine_df = consolidate_variety_geo(wine_df, normalized_geo_df)

  # only keep (variety + geo) which appears more frequently 
  wine_df = trim_by_variety_geo_frequency(wine_df, 30)
  
  # create a dictionary of taste descriptor mapping dataframes for all core tastes
  core_tastes = ['aroma', 'weight', 'sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']
  taste_mapping_set = import_taste_descriptor_mapping(core_tastes)
  
  # for each core taste, map review for every wine into corresponding taste descriptors
  wine_trigram_model = import_wine_phraser()
  for taste in core_tastes:
    taste_df = pd.DataFrame({taste: wine_df['Description'].map(str)})
    taste_df = dask_compute(taste_df, 256, 16, map_review_to_taste_descriptor, taste, normalize_sentence, wine_trigram_model, taste_mapping_set[taste])
    wine_df = pd.concat([wine_df, taste_df], axis=1)
  
  wine_df.to_csv('processed_data/descriptorized_wine_df.csv')


  




   
    


  


  





