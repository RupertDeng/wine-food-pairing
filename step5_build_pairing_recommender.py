from data_importer import import_food_phraser, import_aroma_descriptor_mapping, import_wine_variety_vector_info, import_wine_variety_descriptor_info, import_food_nonaroma_info, import_word2vec_model
from scipy import spatial
from step1_train_word_embedding import normalize_sentence
from step4_prepare_food_data_set import get_food_list_avg_vector


def minmax_scaler(val, min_val, max_val):
  val = min(max(val, min_val), max_val)
  return (val - min_val) / (max_val - min_val)


def standardize_food_similarity(taste, similarity):
  groups = {
    'weight': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
    'sweet': {1: (0, 0.45), 2: (0.45, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
    'acid': {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.7), 4: (0.7, 1)},
    'salt': {1: (0, 0.3), 2: (0.3, 0.55), 3: (0.55, 0.8), 4: (0.8, 1)},
    'piquant': {1: (0, 0.4), 2: (0.4, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
    'fat': {1: (0, 0.4), 2: (0.4, 0.5), 3: (0.5, 0.6), 4: (0.6, 1)},
    'bitter': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.65), 4: (0.65, 1)}
    }
  
  for group, (lower, upper) in groups[taste].items():
    if group == 1 and lower <= similarity <= upper or group != 1 and lower < similarity <= upper:
      return group


def get_standardized_nonaroma_values(taste, avg_food_embedding, food_nonaroma_df):
  avg_taste_vec = food_nonaroma_df.at[taste, 'average_vec']
  farthest = food_nonaroma_df.at[taste, 'farthest']
  closest = food_nonaroma_df.at[taste, 'closest']

  similarity = 1 - spatial.distance.cosine(avg_taste_vec, avg_food_embedding)
  scaled_similarity = minmax_scaler(similarity, closest, farthest)
  standardized_similarity = standardize_food_similarity(taste, similarity)
  
  return (scaled_similarity, standardized_similarity)
  

def retrieve_all_food_attributes(food_list, food_nonaroma_df, core_nonaromas, word2vec):
  food_nonaroma_values = dict()
  avg_food_embedding = get_food_list_avg_vector(food_list)
  for nonaroma in core_nonaromas:
    food_nonaroma_values[nonaroma] = get_standardized_nonaroma_values(nonaroma, avg_food_embedding, food_nonaroma_df)
  return food_nonaroma_values, avg_food_embedding



if __name__ == '__main__':

  # import wine and food processed data, cell with vector data has already been converted to numpy ndarray of float64.
  wine_vector_df = import_wine_variety_vector_info()
  wine_descriptor_df = import_wine_variety_descriptor_info()
  food_nonaroma_df = import_food_nonaroma_info()
  core_nonaromas = ['weight', 'sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']


  # import key utility functions
  food_tokenizer = normalize_sentence
  food_phraser = import_food_phraser()
  aroma_descriptor_mapper = import_aroma_descriptor_mapping()
  word2vec = import_word2vec_model() 

  food_list = ['apple_pie']




  
  
  