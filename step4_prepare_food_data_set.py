from data_importer import import_list_of_foods, import_food_phraser, import_word2vec_model
from step1_train_word_embedding import normalize_sentence
from scipy import spatial
import numpy as np
import pandas as pd

def get_avg_food_vector(foods, word2vec):
  """
  get average food vector from a list of foods
  """
  vec = []
  for f in foods:
    try:
      vec.append(word2vec.wv[f])
    except:
      continue
  if len(vec) > 0:
    return np.average(vec, axis=0)
  else:
    return None
      

if __name__ == '__main__':

  # import the list of food and normalize and phrase each of them
  food_list = import_list_of_foods()
  food_trigram_model = import_food_phraser()
  food_list_normalized = [normalize_sentence(f) for f in food_list]
  food_list_phrased = [food_trigram_model[f] for f in food_list_normalized]

  # get a embedding vector for each food if possible
  word2vec_model = import_word2vec_model()
  food_vectors = dict()
  for food in food_list_phrased:
    vec = get_avg_food_vector(food, word2vec_model)
    if vec is not None:
      food_vectors[' '.join(food)] = vec

  # define the core nonaroma tastes and a list of common words representing each of them
  core_tastes = {
    'weight': ['heavy', 'cassoulet', 'burger', 'full_bodied', 'thick', 'milk', 'fat', 'mincemeat', 'steak', 'bold', 'pizza', 'pasta', 'creamy', 'bread'],
    'sweet': ['sweet', 'sugar', 'cake', 'mango', 'stevia'], 
    'acid': ['acid', 'sour', 'vinegar', 'yoghurt', 'cevich', 'pickle'],
    'salt': ['salty', 'parmesan', 'oyster', 'pizza', 'bacon', 'cured_meat', 'sausage', 'potato_chip'], 
    'piquant': ['spicy', 'pepper', 'mustard', 'paprika', 'curry'], 
    'fat': ['fat', 'fried', 'creamy', 'cassoulet', 'foie_gras', 'buttery', 'sausage', 'brie', 'carbonara', 'cake'], 
    'bitter': ['bitter', 'kale', 'coffee', 'arugula']
    }

  # use core_tastes above to define the average vector for each taste, also calculate vector distance from the average to each food in list
  avg_taste_vecs = dict()
  core_taste_distances = dict()

  for taste, keywords in core_tastes.items():

    avg_vec = get_avg_food_vector(keywords, word2vec_model)
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


  

  

  

  
  






