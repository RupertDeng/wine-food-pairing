import os
import numpy as np
import pandas as pd
import dask
import dask.dataframe as ddf
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer



# download necessary nltk reference data to local folder, only need to be done once.
# nltk.download('punkt', download_dir='/home/rupertd/Projects/wine-food-pairing/nltk_downloads')
# nltk.download('stopwords', download_dir='/home/rupertd/Projects/wine-food-pairing/nltk_downloads')

nltk.data.path.append('/home/rupertd/Projects/wine-food-pairing/nltk_downloads')


# ----------------------------------------------- data import ---------------------------------------------
# import all wine raw data into a dataframe
folder_url = r'raw_data/wine_reviews/'
wine_raw_data = pd.DataFrame()
for file in os.listdir(folder_url):
  file_url = folder_url + '/' + file
  data_to_append = pd.read_csv(file_url, encoding='latin-1', low_memory=False)
  wine_raw_data = pd.concat([wine_raw_data, data_to_append], axis=0, ignore_index=True)

wine_raw_data.drop_duplicates(subset=['Name'], inplace=True)
for geo in ['Subregion', 'Region', 'Province', 'Country']:
  wine_raw_data[geo] = wine_raw_data[geo].apply(lambda g: str(g).strip())

# import food data to a dataframe
food_raw_data = pd.read_csv('raw_data/amazon_food_reviews/Reviews.csv', low_memory=False)


# ---------------------------------------- tokenize sentence with dask's multi-processing ------------------------------------------
def tokenize_sentence_dataframe(df, col):
  new_df = pd.DataFrame({col: df[col].map(sent_tokenize)})
  return new_df.explode(col)

if __name__ == '__main__':

  wine_dataframe = pd.DataFrame().assign(Review=wine_raw_data['Description'].map(str))
  food_dataframe = pd.DataFrame().assign(Review=food_raw_data['Text'].map(str))
  wine_dask_df = ddf.from_pandas(wine_dataframe, npartitions=128)
  food_dask_df = ddf.from_pandas(food_dataframe, npartitions=128)
  wine_sent_tokenized = wine_dask_df.map_partitions(tokenize_sentence_dataframe, 'Review', meta=wine_dask_df).compute(scheduler='processes', num_workers=8)
  food_sent_tokenized = food_dask_df.map_partitions(tokenize_sentence_dataframe, 'Review', meta=food_dask_df).compute(scheduler='processes', num_workers=8)

# --------------------------------------- clean up words in sentence ------------------------------------

stop_words = set(stopwords.words('english'))
punc_table = str.maketrans({key: None for key in string.punctuation})
snowball = SnowballStemmer('english')

def normalize_sentence(sent):
  word_list = word_tokenize(sent)
  normalized_sent = []
  for word in word_list:
    if any(c.isdigit() for c in word):
      continue
    try:
      lowered_word = str(word).lower()
      stemmed_word = snowball.stem(lowered_word)
      no_punc_word = stemmed_word.translate(punc_table)
      if len(no_punc_word) > 1 and no_punc_word not in stop_words:
        normalized_sent.append(no_punc_word)
    except:
      continue
  if normalized_sent:
    return ' '.join(normalized_sent)
  else:
    return np.nan


def normalize_sentence_dataframe(df, col):
  df[col] = df[col].map(normalize_sentence)
  return df

  
if __name__ == '__main__':
  wine_dask_df = ddf.from_pandas(wine_sent_tokenized, npartitions=128)
  fodd_dask_df = ddf.from_pandas(food_sent_tokenized, npartitions=128)
  wine_sent_normalized = wine_dask_df.map_partitions(normalize_sentence_dataframe, 'Review', meta=wine_dask_df).compute(scheduler='processes', num_workers=8)
  food_sent_normalized = food_dask_df.map_partitions(normalize_sentence_dataframe, 'Review', meta=food_dask_df).compute(scheduler='processes', num_workers=8)
  wine_sent_normalized.dropna(inplace=True)
  food_sent_normalized.dropna(inplace=True)
  print(wine_sent_normalized)





  


