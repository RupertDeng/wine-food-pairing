from data_importer import import_descriptorized_wine_data
import pandas as pd
import numpy as np
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
      word_vector = word2vec_model.wv.get_vector(word).reshape(1, 300)
      weighted_word_vector = weighting * word_vector
      weighted_descriptors.append(weighted_word_vector)
    except:
      continue
  if len(weighted_descriptors) > 0:
    return (sum(weighted_descriptors) / len(weighted_descriptors))[0]
  return np.nan
  

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
    avg_taste_vec = np.average(taste_vecs) if not taste_vecs.empty else np.nan
    var_geo_taste_vectors.append(avg_taste_vec)
  return pd.DataFrame({'Variety-Geo': all_variety_geo, col_name: var_geo_taste_vectors})




if __name__ == '__main__':
  wine_df = import_descriptorized_wine_data()
  

