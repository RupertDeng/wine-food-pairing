from data_importer import import_list_of_foods, import_food_phraser, import_word2vec_model, import_aroma_descriptor_mapping
from step1_train_word_embedding import normalize_sentence, find_mapped_descriptor
from scipy import spatial
import numpy as np
import pandas as pd


def get_food_vector(food, text_tokenizer, text_phraser, descriptor_mapper, word2vec):
  food_tokenized = text_tokenizer(food)
  food_phrased = text_phraser[food_tokenized]
  food_descriptorized = [find_mapped_descriptor(part, descriptor_mapper) or part for part in food_phrased]

  vec = []
  for part in food_descriptorized:
    try:
      vec.append(word2vec.wv[part])
    except:
      continue
  
  if len(vec) > 0:
    return np.average(vec, axis=0)
  else:
    return None
  

def get_food_list_avg_vector(food_list, text_tokenizer, text_phraser, descriptor_mapper, word2vec):
  vectors = []
  for food in food_list:
    vec = get_food_vector(food, text_tokenizer, text_phraser, descriptor_mapper, word2vec)
    if vec is not None:
      vectors.append(vec)
  
  if len(vectors) > 0:
    return np.average(vectors, axis=0)
  else:
    return None

      

if __name__ == '__main__':

  # import the list of food, and necessuary utilities for processing: text normalizer, phraser, descrptor_mapper, word2vec_model
  food_list = import_list_of_foods()
  food_text_tokenizer = normalize_sentence
  food_text_phraser = import_food_phraser()
  food_descriptor_mapper = import_aroma_descriptor_mapping()
  word2vec_model = import_word2vec_model()

  food_vectors = dict()
  for food in food_list:
    vec = get_food_vector(food, food_text_tokenizer, food_text_phraser, food_descriptor_mapper, word2vec_model)
    if vec is not None:
      food_vectors[food] = vec

  # define the core nonaroma tastes and a list of common words representing each of them
  core_tastes = {
    'weight': ['heavy', 'cassoulet', 'burger', 'full bodied', 'thick', 'milk', 'fat', 'mince meat', 'steak', 'bold', 'pizza', 'pasta', 'creamy', 'prime rib'],
    'sweet': ['sweet', 'sweet', 'sugar', 'sugar', 'cake', 'mango', 'stevia', 'ice cream'], 
    'acid': ['acid', 'sour', 'vinegar', 'yoghurt', 'cevich', 'pickle', 'cevich'],
    'salt': ['salty', 'salty', 'parmesan', 'oyster', 'pizza', 'bacon', 'bacon', 'cured meat', 'sausage', 'potato chip'], 
    'piquant': ['spicy', 'spicy', 'cayenne pepper', 'mustard', 'paprika', 'curry'], 
    'fat': ['fat', 'fried', 'creamy', 'cassoulet', 'foie gras', 'foie gras', 'buttery', 'sausage', 'brie', 'carbonara', 'cake'], 
    'bitter': ['bitter', 'kale', 'coffee']
    }

  # use core_tastes above to define the average vector for each taste, also calculate vector distance from the average to each food in list
  avg_taste_vecs = dict()
  core_taste_distances = dict()

  for taste, key_food_list in core_tastes.items():

    avg_vec = get_food_list_avg_vector(key_food_list, food_text_tokenizer, food_text_phraser, food_descriptor_mapper, word2vec_model)
    avg_taste_vecs[taste] = avg_vec

    taste_dist = dict()
    for f, v in food_vectors.items():
      similarity = 1 - spatial.distance.cosine(avg_vec, v)
      taste_dist[f] = similarity
    
    core_taste_distances[taste] = taste_dist

  # identify the farthest and closest vector distace among the food list for each taste
  food_nonaroma_info = dict()
  for taste in core_tastes:
    taste_info = dict()
    taste_info['farthest'] = min(core_taste_distances[taste].values())
    taste_info['closest'] = max(core_taste_distances[taste].values())
    taste_info['average_vec'] = avg_taste_vecs[taste]
    food_nonaroma_info[taste] = taste_info

    print(taste, 'farthest', min(core_taste_distances[taste], key=core_taste_distances[taste].get), taste_info['farthest'])
    print(taste, 'closest', max(core_taste_distances[taste], key=core_taste_distances[taste].get), taste_info['closest'])

  food_nonaroma_df = pd.DataFrame(food_nonaroma_info).T
  food_nonaroma_df.to_csv('processed_data/food_nonaroma_df.csv')


  

  

  

  
  






