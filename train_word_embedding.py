from data_importer import import_food_data, import_wine_data, import_descriptor_mapping
import numpy as np
import pandas as pd
import dask
import dask.dataframe as ddf
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models.phrases import Phrases
from gensim.models import word2vec, Word2Vec

# download necessary nltk reference data to local folder, only need to be done once.
# nltk.download('punkt', download_dir='/home/rupertd/Projects/wine-food-pairing/nltk_downloads')
# nltk.download('stopwords', download_dir='/home/rupertd/Projects/wine-food-pairing/nltk_downloads')

nltk.data.path.append('/home/rupertd/Projects/wine-food-pairing/nltk_downloads')
dask.config.set(scheduler='processes', num_workers=16)
stop_words = set(stopwords.words('english'))
punc_table = str.maketrans({key: None for key in string.punctuation})
snowball = SnowballStemmer('english')
word2vec.FAST_VERSION = 1


def tokenize_sentence_dataframe(df, col):
  '''
  Tokenize sentence on each row in the dataframe at column 'col', and spread the resulting list of sentence tokens from each row into multiple rows.
  Return a new dataframe with one sentence token per row.
  '''
  new_df = pd.DataFrame({col: df[col].map(sent_tokenize)})
  return new_df.explode(col)


def normalize_sentence(sent):
  """
  Normalize the words in one sentence by removing numbers, lowercasing, stemming, and removing puncuation.
  Return a list of normalized words in the sentence. If not words left, return numpy nan object.
  """
  word_list = word_tokenize(sent)
  normalized_word_list = []
  for word in word_list:
    if any(c.isdigit() for c in word):
      continue
    try:
      lowered_word = str(word).lower()
      stemmed_word = snowball.stem(lowered_word)
      no_punc_word = stemmed_word.translate(punc_table)
      if len(no_punc_word) > 1 and no_punc_word not in stop_words:
        normalized_word_list.append(no_punc_word)
    except:
      continue
  if normalized_word_list:
    return normalized_word_list
  else:
    return np.nan


def normalize_sentence_dataframe(df, col):
  '''
  Use the normalize_sentence function to clean up each row in the data frame at column 'col'.
  Return the data frame after change, with each row having a list of normalized word tokens.
  '''
  df[col] = df[col].map(normalize_sentence)
  return df


def find_mapped_descriptor(word, mapping):
  try:
    descriptor = mapping.at[word, 'level_3']
    return str(descriptor)
  except KeyError:
    return word



if __name__ == '__main__':

  # import raw data
  wine_raw_data = import_wine_data()
  food_raw_data = import_food_data()

  # extract key review texts for wine and food
  wine_sent_raw = pd.DataFrame().assign(Text=wine_raw_data['Description'].map(str))
  food_sent_raw = pd.DataFrame().assign(Text=food_raw_data['Text'].map(str))
  
  # tokenize sentence
  wine_sent_tokenized = ddf.from_pandas(wine_sent_raw, npartitions=256).map_partitions(tokenize_sentence_dataframe, 'Text', meta=wine_sent_raw).compute()
  food_sent_tokenized = ddf.from_pandas(food_sent_raw, npartitions=256).map_partitions(tokenize_sentence_dataframe, 'Text', meta=food_sent_raw).compute()
  # print(wine_sent_tokenized)
  # print(food_sent_tokenized)

  # normalize words in sentence and remove empty sentence
  wine_sent_normalized = ddf.from_pandas(wine_sent_tokenized, npartitions=256).map_partitions(normalize_sentence_dataframe, 'Text', meta=wine_sent_tokenized).compute()
  food_sent_normalized = ddf.from_pandas(food_sent_tokenized, npartitions=256).map_partitions(normalize_sentence_dataframe, 'Text', meta=food_sent_tokenized).compute()
  wine_sent_normalized.dropna(inplace=True)
  food_sent_normalized.dropna(inplace=True)
  # print(wine_sent_normalized)
  # print(food_sent_normalized)

  # use the whole review text corpus to train gensim bigram and tri-gram models
  wine_bigram_model = Phrases(wine_sent_normalized['Text'], min_count=100)
  wine_bigrams = [wine_bigram_model[sent] for sent in wine_sent_normalized['Text']]
  wine_trigram_model = Phrases(wine_bigrams, min_count=50)
  wine_sent_phrased = [wine_trigram_model[sent] for sent in wine_bigrams]

  food_bigram_model = Phrases(food_sent_normalized['Text'], min_count=100)
  food_bigrams = [food_bigram_model[sent] for sent in food_sent_normalized['Text']]
  food_trigram_model = Phrases(food_bigrams, min_count=50)
  food_sent_phrased =[food_trigram_model[sent] for sent in food_bigrams]

  wine_trigram_model.save('trained_models/wine_trigram_model.pkl')
  food_trigram_model.save('trained_models/food_trigram_model.pkl')

  # print(len(wine_sent_phrased))
  # print(wine_sent_phrased[:10])
  # print(len(food_sent_phrased))
  # print(food_sent_phrased[:10])

  # map common used words for descriping wine to a set of normalized descriptors
  descriptor_mapping = import_descriptor_mapping()
  wine_sent_mapped = [[find_mapped_descriptor(word, descriptor_mapping) for word in sent] for sent in wine_sent_phrased]

  # do the same mapping for food, but skip the non-aroma descriptors
  aroma_descriptor_mapping = descriptor_mapping.loc[descriptor_mapping['type'] == 'aroma']
  food_sent_mapped = [[find_mapped_descriptor(word, aroma_descriptor_mapping) for word in sent] for sent in food_sent_phrased]


  wine_food_sent_combined = wine_sent_mapped + food_sent_mapped
  wine_food_word2vec_model = Word2Vec(wine_food_sent_combined, vector_size=300, min_count=8, workers=8, epochs=15)
  # print(wine_food_word2vec_model)

  wine_food_word2vec_model.save('trained_models/wine_food_word2vec_model.pkl')





  
  





  


