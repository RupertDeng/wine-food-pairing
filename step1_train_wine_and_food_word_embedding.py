from data_importer import import_food_data, import_wine_data, import_descriptor_mapping, import_aroma_descriptor_mapping
from dask_multiprocessing import dask_compute
import numpy as np
import pandas as pd
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


def multi_gram_phrase_conversion(df, col, model):
  """
  convert each sentence row in df[col] based on multi-gram model 'model'
  """
  conversion = lambda sent: model[sent]
  df[col] = df[col].map(conversion)
  return df


def find_mapped_descriptor(word, mapping):
  """
  find corresponding descriptor for 'word' in mapping, return empty string if not exist
  """
  try:
    return str(mapping.at[word, 'combined']).strip()
  except:
    return ''


def mapped_descriptor_conversion(df, col, mapping):
  """
  convert each sentence row in df[col] based on the descriptor mapping. if not descriptor exists, simply use the word itself.
  """
  conversion = lambda sent: [find_mapped_descriptor(word, mapping) or word for word in sent]
  df[col] = df[col].map(conversion)
  return df


if __name__ == '__main__':

  # import raw data
  wine_raw_data = import_wine_data()
  food_raw_data = import_food_data()

  # extract key review texts for wine and food
  wine_sent_raw = pd.DataFrame().assign(Text=wine_raw_data['Description'].map(str))
  food_sent_raw = pd.DataFrame().assign(Text=food_raw_data['Text'].map(str))

  print('------------------- raw data imported --------------------')
  print(wine_sent_raw)
  print(food_sent_raw)
  print('\n')
  
  # tokenize sentence
  wine_sent_tokenized = dask_compute(wine_sent_raw, 256, 16, tokenize_sentence_dataframe, 'Text')
  food_sent_tokenized = dask_compute(food_sent_raw, 256, 16, tokenize_sentence_dataframe, 'Text')

  print('------------------- sentence tokenized --------------------')
  print(wine_sent_tokenized)
  print(food_sent_tokenized)
  print('\n')

  # normalize words in sentence and remove empty sentence
  wine_sent_normalized = dask_compute(wine_sent_tokenized, 256, 16, normalize_sentence_dataframe, 'Text')
  food_sent_normalized = dask_compute(food_sent_tokenized, 256, 16, normalize_sentence_dataframe, 'Text')
  wine_sent_normalized.dropna(inplace=True)
  food_sent_normalized.dropna(inplace=True)

  print('------------------- word normalized --------------------')
  print(wine_sent_normalized)
  print(food_sent_normalized)
  print('\n')

  # use the whole review text corpus to train gensim bigram and tri-gram models
  wine_bigram_model = Phrases(wine_sent_normalized['Text'], min_count=100, threshold=10)
  wine_bigrams = dask_compute(wine_sent_normalized, 256, 16, multi_gram_phrase_conversion, 'Text', wine_bigram_model)
  wine_trigram_model = Phrases(wine_bigrams['Text'], min_count=50, threshold=10)
  wine_sent_phrased = dask_compute(wine_bigrams, 256, 16, multi_gram_phrase_conversion, 'Text', wine_trigram_model)

  food_bigram_model = Phrases(food_sent_normalized['Text'], min_count=100, threshold=1)
  food_bigrams = dask_compute(food_sent_normalized, 256, 16, multi_gram_phrase_conversion, 'Text', food_bigram_model)
  food_trigram_model = Phrases(food_bigrams['Text'], min_count=50, threshold=1)
  food_sent_phrased = dask_compute(food_bigrams, 256, 16, multi_gram_phrase_conversion, 'Text', food_trigram_model)

  wine_trigram_model.save('trained_models/wine_trigram_model.pkl')
  food_trigram_model.save('trained_models/food_trigram_model.pkl')


  print('------------------- sentence converted with multi-gram phrases --------------------')
  print(wine_sent_phrased)
  print(food_sent_phrased)
  print('\n')

  # map common used words for descriping wine to a set of normalized descriptors
  descriptor_mapping = import_descriptor_mapping()
  wine_sent_mapped = dask_compute(wine_sent_phrased, 256, 16, mapped_descriptor_conversion, 'Text', descriptor_mapping)

  # do the same mapping for food, but skip the non-aroma descriptors
  aroma_descriptor_mapping = import_aroma_descriptor_mapping()
  food_sent_mapped = dask_compute(food_sent_phrased, 256, 16, mapped_descriptor_conversion, 'Text', aroma_descriptor_mapping)
  
  print('------------------- sentence converted with descriptor mapping --------------------')
  print(wine_sent_mapped)
  print(food_sent_mapped)
  print('\n')

  wine_food_sent_combined = list(wine_sent_mapped['Text']) + list(food_sent_mapped['Text'])
  wine_food_word2vec_model = Word2Vec(wine_food_sent_combined, vector_size=300, min_count=8, workers=16, epochs=15)
  
  print('------------------- word2vec model trained --------------------')
  print(wine_food_word2vec_model)

  wine_food_word2vec_model.save('trained_models/wine_food_word2vec_model.pkl')





  
  





  


