from data_importer import import_wine_data, import_variety_mapping, import_normalized_geo_data, import_taste_descriptor_mapping
import pandas as pd


def cleanup_variety(df, mapping):
  df['Variety'] = df['Variety'].map(lambda v: mapping.get(v, v))
  return df


def replace_nan_with_none(value):
  if str(value) in ['0', 'nan']:
    return 'none'
  else:
    return value


def cleanup_geo(df):
  order_of_geographies = ['Subregion', 'Region', 'Province', 'Country']
  for geo in order_of_geographies:
    df[geo] = df[geo].map(replace_nan_with_none)
  df.loc[:, order_of_geographies].fillna('none', inplace=True)
  return df


def consolidate_variety_geo(df, mapping):
  merge_columns = ['Variety', 'Country', 'Province', 'Region', 'Subregion']
  df = pd.merge(left=df, right=mapping, left_on=merge_columns, right_on=merge_columns)
  df = df[['Name', 'Variety', 'geo_normalized', 'Description']]
  return df


def trim_by_variety_geo_frequency(df, freq):
  trim_columns = ['Variety', 'geo_normalized']
  variety_geos = df.groupby(trim_columns).size().reset_index().rename(columns={0: 'count'})
  variety_geos = variety_geos.loc[variety_geos['count'] > freq]
  trimmed_df = pd.merge(left=df, right=variety_geos, left_on=trim_columns, right_on=trim_columns).drop(['count'], axis=1).dropna()
  return trimmed_df







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
  
  core_tastes = ['aroma', 'weight', 'sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']
  taste_mapping_set = import_taste_descriptor_mapping(core_tastes)

  
  
  
  


  


  





