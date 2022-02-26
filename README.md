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





