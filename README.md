# Wine-Food-Pairing, a Python NLP application
### This is a re-creation of Roald Schuring's original wine-food-pairing code in jupyter notebook, mainly for learning purpose. You can check his repository in this link: [Wine Food Pairing Jupyter Notebook](https://github.com/RoaldSchuring/wine_food_pairing).
### I have restructured his code logic into a set of more debug-friendly modulized scripts, and more importantly, incorporated dask multi-processing to reduce the time of processing huge pandas dataframe. It is quite a fun journey to learn through all the NLP essentials, plus some cool knowledge on wines. Below are the key notes on my version, as a record.
<br/>

## 1) Raw data needed
- Most of the raw and reference data are kept the same as what Roald has used. Details are explained in his repository.
- Wine review raw data scraped from www.winemag.com: [link](https://www.kaggle.com/roaldschuring/wine-reviews)
- Amazon fine food review raw data from kaggle: [link](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- <mark>The above two raw data set are saved under '/raw_data'.</mark>.
- Wine normalized variety-geo reference list prepared by Roald: [link](https://github.com/RoaldSchuring/wine_food_pairing/blob/master/varieties_all_geos_normalized.csv)
- Core taste descriptor map to normalize review terms: [link](https://github.com/RoaldSchuring/wine_food_pairing/blob/master/descriptor_mapping_tastes.csv)
- A comprehensive list of food names: [link](https://github.com/RoaldSchuring/wine_food_pairing/blob/master/list_of_foods.csv)
- <mark>The above three referece data are saved under '/refereces'.</mark>

