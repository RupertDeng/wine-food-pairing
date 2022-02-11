from data_importer import import_descriptorized_wine_data, import_word2vec_model
import pandas as pd
import numpy as np
from dask_multiprocessing import dask_compute
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import Counter


def convert_descriptor_to_vector(descriptors, tfidf_weighting, word2vec_model):
  """
  for a list of descriptors, convert each word to a vector using tfidf and word2vec.
  then average all vectors to get one final vector.
  """
  weighted_descriptors = []
  for word in descriptors.split(' '):
    if not word: continue
    try:
      weighting = tfidf_weighting[word]
      word_vector = word2vec_model.wv[word]
      weighted_word_vector = weighting * word_vector
      weighted_descriptors.append(weighted_word_vector)
    except:
      continue
  if len(weighted_descriptors) > 0:
    return np.average(weighted_descriptors, axis=0)
  return np.nan
  

def vectorize_taste_dataframe(df, col, tfidf_weighting, word2vec_model):
  """
  convert each row of df (containing a list of descriptors) into one vector
  """
  df[col] = df[col].map(lambda sent: convert_descriptor_to_vector(sent, tfidf_weighting, word2vec_model))
  return df


def get_limited_taste_scalar(variety_df, taste):
  """
  for limited taste, simply get the percentage of reviews mentioning this taste for each variety
  """
  total_review = variety_df[taste].size
  mentioned_review = variety_df[taste].loc[variety_df[taste] != ''].size
  return mentioned_review / total_review


def get_most_freq_aroma_descriptor(variety_df):
  """
  for aroma descriptors, get the 50 most common words and their percentage in all descriptors
  """
  all_descriptors = []
  for desc in variety_df['aroma_descriptor']:
    if desc == '': continue
    all_descriptors.extend(desc.split(' '))
  desc_freqs = Counter(all_descriptors)
  most_common_desc = desc_freqs.most_common(50)
  return [(desc[0], '{:.2f}'.format(desc[1] / len(variety_df) * 100)) for desc in most_common_desc]


def get_variety_vectors_descriptors(variety_geo, wine_df, core_tastes, limited_taste):
  """
  for each variety-geo combination, return a row of data including the most_common aroma descriptors, and average vector for each taste.
  limited-taste will be treated differently, see above.
  """
  variety_df = wine_df.loc[(wine_df['Variety'] == variety_geo[0]) & (wine_df['geo_normalized'] == variety_geo[1])]
  output_df = pd.DataFrame({'Variety': [variety_geo[0]], 'Geo': [variety_geo[1]]})
  for taste in core_tastes:
    if taste in limited_taste:
      col_name = taste + ' scalar'
      output_df[col_name] = [get_limited_taste_scalar(variety_df, taste)]
    else:
      if taste == 'aroma':
        desc_col = taste + ' descriptors'
        output_df[desc_col] = [get_most_freq_aroma_descriptor(variety_df)]
      vec_col = taste + ' vector'
      output_df[vec_col] = [np.average(list(variety_df[taste]), axis=0)]
  return output_df


def normalize_nonaroma_scalar(wine_df, taste):
  col_name = taste + ' scalar'
  max_value = wine_df[col_name].max()
  min_value = wine_df[col_name].min()
  df[col_name] = df[col_name].map(lambda x: (x - min_value) / (max_value - min_value))
  return df


if __name__ == '__main__':
  
  # import descriptoried wine dataframe, both the aroma and nonaroma attributes have been standardized against a set of descriptors
  # for aroma, weight, sweet, and bitter, the empty cells can be replace with the average vector from all reivews of all wines, since the descriptors typically cover full spectrum
  # for salt, piquant, and fat, it is more tricky due to limited descriptors, instead, we can use the percentage reviews mentioning this attribute as its scalar.
  # the goal is to get an average aroma vec and a set of nonaroma scalar for each wine

  wine_df = import_descriptorized_wine_data()
  word2vec_model = import_word2vec_model()
  core_tastes = ['aroma', 'weight', 'sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']
  limited_tastes = ['salt', 'piquant', 'fat']

  # for each taste, convert the descriptors to a weighted word-embedding vector
  for taste in core_tastes:
    if taste in limited_tastes: continue    # only process non-limited tastes in this block
    taste_words = wine_df[taste]
    if taste == 'aroma': wine_df['aroma_descriptor'] = list(taste_words)  # store extra aroa_descriptor column to get most frequent descriptor for each wie
    vectorizer = TfidfVectorizer()
    V = vectorizer.fit(taste_words)
    tfidf_weighting_dict = dict(zip(V.get_feature_names_out(), V.idf_))
    wine_df = dask_compute(wine_df, 256, 16, vectorize_taste_dataframe, taste, tfidf_weighting_dict, word2vec_model)
    
    # for empty cell of non-limited tastes, simply fill in with the average vector from all views for that taste
    non_zero_vec = list(wine_df[taste].dropna())
    avg_vec = np.average(non_zero_vec, axis=0)
    wine_df[taste] = [avg_vec if 'numpy' not in str(type(r)) else r for r in wine_df[taste]]

  # now, we need to converge all data for each variety-geo combination
  # for limited-tastes, get a frequency scalar; and for other tastes, get an average vector
  variety_geos = sorted(set(zip(wine_df['Variety'], wine_df['geo_normalized'])))
  wine_variety_df = pd.DataFrame()
  for v_g in variety_geos:
    row_for_variety = get_variety_vectors_descriptors(v_g, wine_df, core_tastes, limited_tastes)
    wine_variety_df = pd.concat([wine_variety_df, row_for_variety], axis=0, ignore_index=True)

  # for non-aroma taste of each wine which still has a vector, we will do a PCA reduction to scalar, to align with limited tastes
  for taste in core_tastes:
    if taste in limited_tastes or taste == 'aroma':
      continue
    col_name = taste + ' vector'
    pca = PCA(1)
    wine_variety_df[col_name] = pca.fit_transform(list(wine_variety_df[col_name]))
    wine_variety_df.rename(columns = {col_name: taste + ' scalar'}, inplace=True)


  # store most common descriptors in a separate dataframe with one descriptor per line
  wine_variety_descriptor_df = wine_variety_df[['Variety', 'Geo', 'aroma descriptors']].explode('aroma descriptors', ignore_index=True)
  wine_variety_descriptor_df[['descriptor', 'frequency']] = wine_variety_descriptor_df['aroma descriptors'].to_list()
  wine_variety_descriptor_df.drop('aroma descriptors', axis=1, inplace=True)

  wine_variety_vector_df = wine_variety_df.drop('aroma descriptors', axis=1)

  # for the nonaroma scalars, sweet/salt/piquant/fat are in order, weight/acid/bitter need to be flipped to match common sense (larger value means more)
  for taste in ['weight', 'acid', 'bitter']:
    col_name = taste + ' scalar'
    wine_variety_vector_df[col_name] = wine_variety_vector_df[col_name].map(lambda x: -x)


  # for the last step, normalize all the nonaroma scalars into [0, 1]
  for taste in core_tastes:
    if taste == 'aroma': continue
    wine_variety_vector_df = normalize_nonaroma_scalar(wine_variety_vector_df, taste)

  
  wine_variety_vector_df.to_csv('processed_data/wine_variety_vector.csv')
  wine_variety_descriptor_df.to_csv('processed_data/wine_variety_aroma_descriptor.csv')
  
  

  


  

  

  

  











  

