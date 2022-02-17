from webbrowser import get
from data_importer import import_food_phraser, import_aroma_descriptor_mapping, import_wine_variety_vector_info, import_wine_variety_descriptor_info, import_food_nonaroma_info, import_word2vec_model
from scipy import spatial
from step1_train_word_embedding import normalize_sentence
from step4_prepare_food_data_set import get_food_list_avg_vector
from step5_define_pairing_rules import nonaroma_ruling, congruent_or_contrasting


def minmax_scaler(val, min_val, max_val):
  val = min(max(val, min_val), max_val)
  return (val - min_val) / (max_val - min_val)


def standardize_food_similarity(taste, similarity):
  groups = {
    'weight': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
    'sweet': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
    'acid': {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.68), 4: (0.68, 1)},
    'salt': {1: (0, 0.42), 2: (0.42, 0.55), 3: (0.55, 0.72), 4: (0.72, 1)},
    'piquant': {1: (0, 0.5), 2: (0.5, 0.61), 3: (0.61, 0.8), 4: (0.8, 1)},
    'fat': {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.65), 4: (0.65, 1)},
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
  scaled_similarity = minmax_scaler(similarity, farthest, closest)
  standardized_similarity = standardize_food_similarity(taste, scaled_similarity)
  
  return (scaled_similarity, standardized_similarity)
  

def retrieve_all_food_attributes(food_list, food_nonaroma_df, core_nonaromas, text_tokenizer, text_phraser, descriptor_mapper, word2vec):
  food_nonaroma_values = dict()
  avg_food_embedding = get_food_list_avg_vector(food_list, text_tokenizer, text_phraser, descriptor_mapper, word2vec)
  for nonaroma in core_nonaromas:
    value = get_standardized_nonaroma_values(nonaroma, avg_food_embedding, food_nonaroma_df)
    if nonaroma == 'weight': food_weight = value
    else: food_nonaroma_values[nonaroma] = value
  return food_nonaroma_values, food_weight, avg_food_embedding


def standardize_wine_nonaroma_scalar(taste, wine_scalar):
  groups = {
    'weight': {1: (0, 0.3), 2: (0.3, 0.55), 3: (0.55, 0.75), 4: (0.75, 1)},
    'sweet': {1: (0, 0.25), 2: (0.25, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
    'acid': {1: (0, 0.6), 2: (0.6, 0.8), 3: (0.8, 0.95), 4: (0.95, 1)},
    'salt': {1: (0, 0.1), 2: (0.1, 0.25), 3: (0.25, 0.45), 4: (0.45, 1)},
    'piquant': {1: (0, 0.1), 2: (0.1, 0.3), 3: (0.3, 0.55), 4: (0.55, 1)},
    'fat': {1: (0, 0.1), 2: (0.1, 0.25), 3: (0.25, 0.55), 4: (0.55, 1)},
    'bitter': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.75), 4: (0.75, 1)}
    }

  for group, (lower, upper) in groups[taste].items():
    if group == 1 and lower <= wine_scalar <= upper or group != 1 and lower < wine_scalar <= upper:
      return group


def sort_by_aroma_similarity(df, food_vector):
  df['aroma distance'] = df['aroma vector'].apply(lambda v: spatial.distance.cosine(v, food_vector))
  df.sort_values(by=['aroma distance'], ascending=True, inplace=True)
  return df


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


  # standardize wine nonaroma scalar to a scale of 1 to 4
  for taste in core_nonaromas:
    col_name = taste + ' scalar'
    wine_vector_df[col_name] = wine_vector_df[col_name].map(lambda scalar: standardize_wine_nonaroma_scalar(taste, scalar))

  
  food_list = ['steak']
  food_nonaroma_values, food_weight, food_avg_vector = retrieve_all_food_attributes(food_list, food_nonaroma_df, core_nonaromas, food_tokenizer, food_phraser, aroma_descriptor_mapper, word2vec)

  wine_recommendations = nonaroma_ruling(wine_vector_df, food_nonaroma_values, food_weight)
  wine_recommendations = congruent_or_contrasting(wine_recommendations, food_nonaroma_values)
  wine_recommendations = sort_by_aroma_similarity(wine_recommendations, food_avg_vector)
  print(wine_recommendations)
  
  
  