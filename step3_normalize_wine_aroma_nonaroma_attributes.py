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
  return ''
  

def vectorize_taste_dataframe(df, col, tfidf_weighting, word2vec_model):
  """
  convert each row of df (containing a list of descriptors) into one vector
  """
  df[col] = df[col].map(lambda sent: convert_descriptor_to_vector(sent, tfidf_weighting, word2vec_model))
  return df


def get_variety_geo_taste_vectors(wine_df, all_variety_geo, taste):
  col_name = f'{taste} vector'
  var_geo_taste_vectors = []
  for vg in all_variety_geo:
    subset_df = wine_df.loc[(wine_df['Variety'] == vg[0]) & (wine_df['geo_normalized'] == vg[1])]
    taste_vecs = subset_df[col_name].dropna()
    # avg_taste_vec = np.average(taste_vecs) if not taste_vecs.empty else np.nan
    var_geo_taste_vectors.append(avg_taste_vec)
  return pd.DataFrame({'Variety-Geo': all_variety_geo, col_name: var_geo_taste_vectors})




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
    
    non_zero_vec = list(wine_df[taste].replace('', np.nan).dropna())
    avg_vec = np.average(non_zero_vec, axis=0)
    wine_df[taste] = [avg_vec if isinstance(r, str) else r for r in wine_df[taste]]

  

  











  

