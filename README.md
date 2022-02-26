# Wine-Food-Pairing, a Python NLP application
### This is a re-creation of Roald Schuring's original wine-food-pairing code in jupyter notebook, mainly for learning purpose. You can check his repository in this link: [Wine Food Pairing Jupyter Notebook](https://github.com/RoaldSchuring/wine_food_pairing).
### I have restructured his code logic into a set of more debug-friendly modulized scripts, and more importantly, incorporated dask multi-processing to reduce the time of processing huge pandas dataframe. It is quite a fun journey to learn through all the NLP essentials, plus some cool knowledge on wines. Below are the key notes on my version, as a record.
<br/>

## 1) Raw data needed
- Most of the raw and reference data are kept the same as what Roald has used. Details are explained in his repository.
- The below two raw data sets are saved under '/raw_data'.
  - Wine review raw data scraped from www.winemag.com: [link](https://www.kaggle.com/roaldschuring/wine-reviews)
  - Amazon fine food review raw data from kaggle: [link](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- The below three referece data are saved under '/references'.
  - Wine normalized variety-geo reference list prepared by Roald: [link](https://github.com/RoaldSchuring/wine_food_pairing/blob/master/varieties_all_geos_normalized.csv)
  - Core taste descriptor map to normalize review terms: [link](https://github.com/RoaldSchuring/wine_food_pairing/blob/master/descriptor_mapping_tastes.csv)
  - A comprehensive list of food names: [link](https://github.com/RoaldSchuring/wine_food_pairing/blob/master/list_of_foods.csv)

## 2) Dask multi-processing
- Dask is a very powerful and handy tool to accelarate data processing on large-sized pandas dataframe.
- Under the hood, it splits the dataframe by row into multiple partitions, and process them in parallel with python multi-thread/multi-processing support, then merge the partitions back together.
- The data being processed here is in the range of 100k ~ 500k rows. With dask's help, the script typically finishes each task within 10min on my Ryzen 5800x CPU.
- Below is the dask multi-processing code. It will take original pandas dataframe and return a processed pandas dataframe. 
```
# npartitions: the number of partitions you would like to split into.
# nworkers: the number of processes to be employed on processing.
# for the usage of 'meta' and 'align_dataframes', please refer to dask's documentation

dask.dataframe
.from_pandas(pandas_dataframe, npartitions=npartitions)
.map_partitions(process_function, meta=pandas_dataframe, align_dataframes=False)
.compute(scheduler='processes', num_workers=nworkers)
```


## 3) Step1_train_wine_and_food_word_embedding
- This script will first process raw wine and food review sentences with `nltk sent_tokenize, word_tokenize, and SnowballStemmer` to convert words to normalized terms.
- Then the wine and food corpora will be individually sent through `gensim Phrases` to capture multi-gram phrases frequently mentioned. I found this one can be a bit tricky with the threshold setting. Sometimes finding two many phrases might be bad for downstream process since it can hide useful single words. Also gensim phraser seems not to have a multi-processing enabler, it takes relatively long time.
- The next step is to `map some important terms into standard descriptors`, for both wine and food corpora. This is essential to distinguishing wine tastes, as well as linking wine and food data set properly.
- The last step in this script is to train combined processed wine and food corpora in `gensim Word2Vec` to get a word-to-vector model. Word2vec library does have a multi-worker option.
- The trained model is saved in '/trained_models/'.

## 4) Step2_prepare_wine_data_set
- Now the wine_data set is being trimmed by wine variety and geography, only keeping the relatively frequent ones.
- And the resulting dataframe is re-processed through tokenizer, phraser and descriptor mapper under each core tastes in ['aroma', 'weight', 'sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']. This is to extract key information from every wine review sentences.
- The processed wine data is saved to csv under '/processed_data'.







