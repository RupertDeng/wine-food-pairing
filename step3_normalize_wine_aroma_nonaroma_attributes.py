from data_importer import import_descriptorized_wine_data, import_word2vec_model
import pandas as pd
import numpy as np
from dask_multiprocessing import dask_compute
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


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
  total_review = variety_df[taste].size
  mentioned_review = variety_df[taste].loc[variety_df[taste] != ''].size
  return mentioned_review / total_review

def get_most_freq_descriptor(variety_df, taste):
  pass


def get_variety_vectors_descriptors(variety_geo, wine_df, core_tastes, limited_taste):
  variety_df = wine_df.loc[(wine_df['Variey'] == variety_geo[0]) & (wine_df['geo_normalized'] == variety_geo[1])]
  output_df = pd.DataFrame({'Variety': [variety_geo[0]], 'Geo': [variety_geo[1]]})
  for taste in core_tastes:
    if taste in limited_taste:
      col_name = taste + ' scalar'
      output_df[col_name] = [get_limited_taste_scalar(variety_df, taste)]
    else:
      vec_col = taste + ' vector'
      output_df[vec_col] = [np.average(list(variety_df[taste]), axis=0)]
      if taste == 'aroma':
        desc_col = taste + ' descriptors'
        output_df[desc_col] = [get_most_freq_descriptor(variety_df, taste)]
  return output_df




if __name__ == '__main__':
  
  # import descriptoried wine dataframe, both the aroma and nonaroma attributes have been standardized against a set of descriptors
  # for aroma, weight, sweet, and bitter, the empty cells can be replace with the average vector from all reivews of all wines, since the descriptors typically cover full spectrum
  # for salt, piquant, and fat, it is more tricky due to limited descriptors, instead, we can use the percentage reviews mentioning this attribute as its scalar.
  # the goal is to get an average aroma vec and a set of nonaroma scalar for each wine

  wine_df = import_descriptorized_wine_data()
  word2vec_model = import_word2vec_model()
  core_tastes = ['aroma', 'weight', 'sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']
  limited_tastes = ['salt', 'piquant', 'fat']

  for taste in core_tastes:
    if taste in limited_tastes: continue    # only process non-limited tastes in this block
    taste_words = wine_df[taste]
    if taste == 'aroma': wine_df['aroma_descriptor'] = list(taste_words)  # store extra aroa_descriptor column to get most frequent descriptor for each wie
    vectorizer = TfidfVectorizer()
    V = vectorizer.fit(taste_words)
    tfidf_weighting_dict = dict(zip(V.get_feature_names_out(), V.idf_))
    wine_df = dask_compute(wine_df, 256, 16, vectorize_taste_dataframe, taste, tfidf_weighting_dict, word2vec_model)
    
    non_zero_vec = list(wine_df[taste].dropna())
    avg_vec = np.average(non_zero_vec, axis=0)
    wine_df[taste] = [avg_vec if 'numpy' not in str(type(r)) else r for r in wine_df[taste]]

  


  

  

  

  











  

