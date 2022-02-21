from data_importer import import_food_phraser, import_aroma_descriptor_mapping, import_wine_variety_vector_info, import_wine_variety_descriptor_info, import_food_nonaroma_info, import_word2vec_model
from scipy import spatial
from step1_train_wine_and_food_word_embedding import normalize_sentence
from step4_prepare_food_data_set import get_food_list_avg_vector
from step5_define_pairing_rules import nonaroma_ruling, congruent_or_contrasting_pairing
import pandas as pd
from step6_make_visualization_tool import plot_wine_recommendations


def minmax_scaler(val, min_val, max_val):
  val = min(max(val, min_val), max_val)
  return (val - min_val) / (max_val - min_val)


def standardize_food_similarity(taste, similarity):
  """
  standardize food similarity for each taste to a 1-4 level, based on min-max scaling.
  """
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
  """
  This is the key function to extract food nonaroma values based on similarity of food_embedding with average vector for each taste.
  
  args:
  - taste: the nonaroma taste value to be exacted
  - avg_food_embedding: averge vector for the list of food being examined
  - food_nonaroma_df: the reference dataframe containing average vector for each taste, with taste as the row index, and 'average_vec', 'farthest', 'closest' as the column index
  """
  avg_taste_vec = food_nonaroma_df.at[taste, 'average_vec']
  farthest = food_nonaroma_df.at[taste, 'farthest']
  closest = food_nonaroma_df.at[taste, 'closest']

  similarity = 1 - spatial.distance.cosine(avg_taste_vec, avg_food_embedding)
  scaled_similarity = minmax_scaler(similarity, farthest, closest)
  standardized_similarity = standardize_food_similarity(taste, scaled_similarity)
  
  return (scaled_similarity, standardized_similarity)
  

def retrieve_all_food_attributes(food_list, food_nonaroma_df, core_nonaromas, text_tokenizer, text_phraser, descriptor_mapper, word2vec):
  """
  function to get all key attribute values from a list of food, including weight, nonaroma values, and average embedding vector
  
  args:
  - food_list: a list of food strings being examined
  - food_nonaroma_df: a reference dataframe containing average/farthest/closest vector thoughout all general foods for each taste
  - core_nonaromas: the list of nonaroma tastes whose values are to be extracted
  - text_tokenizer: utility function to tokenize and normalize food string
  - text_phraser: utility function to convert tokenized food string to standard phrases
  - descriptor_mapper: a reference descriptor_mapping dataframe which will be used to convert some aroma word in food string to uniformed word
  - word2vec: the word2vec model trained with all processed wine and food vocabulary
  """
  food_nonaroma_values = dict()
  avg_food_embedding = get_food_list_avg_vector(food_list, text_tokenizer, text_phraser, descriptor_mapper, word2vec)
  for nonaroma in core_nonaromas:
    value = get_standardized_nonaroma_values(nonaroma, avg_food_embedding, food_nonaroma_df)
    if nonaroma == 'weight': food_weight = value
    else: food_nonaroma_values[nonaroma] = value
  return food_nonaroma_values, food_weight, avg_food_embedding


def standardize_wine_nonaroma_scalar(taste, wine_scalar):
  """
  standardize the wine nonaroma scalars to a scale of 1 - 4.
  """
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


def pick_wines_by_geo(wine_df, geo_picker):
  """
  function to filter wine_df to contain only certain countries
  - geo_picker: a list of country strings to be included in filtered dataframe
  """
  if geo_picker == []:
    return wine_df
  picking = lambda x: any(geo in x for geo in geo_picker)
  return wine_df.loc[wine_df['Geo'].apply(picking) == True]


def sort_by_aroma_similarity(df, food_vector):
  """
  simple sorting function to order the wine dataframe based on the aroma similarity to food vector
  """
  df['aroma distance'] = df['aroma vector'].apply(lambda v: spatial.distance.cosine(v, food_vector))
  df.sort_values(by=['aroma distance'], ascending=True, inplace=True)
  return df


def get_descriptor_similarity(descriptor, food_vec, word2vec):
  """
  simple function to get nonaroma similarity to food vector for certain aroam descriptor
  """
  descriptor_vec = word2vec.wv[descriptor]
  return 1 - spatial.distance.cosine(descriptor_vec, food_vec)


def get_most_impactful_descriptors(wine_descriptor_df, variety, geo, food_vec, word2vec):
  """
  function to get the 5 most impactful descriptors for a specific variety-geo wine
  sorted by descriptor's similarity with food vector, then frequency
  """
  df = wine_descriptor_df.loc[(wine_descriptor_df['Variety'] == variety) & (wine_descriptor_df['Geo'] == geo)]
  df['similarity'] = df['descriptor'].map(lambda d: get_descriptor_similarity(d, food_vec, word2vec))
  df.sort_values(['similarity', 'frequency'], ascending=False, inplace=True)
  df = df.head(5)
  return list(df['descriptor'])


def merge_congruent_and_contrasting(congruent, contrasting):
  """
  merge congruent and contrasting recommendation dataframe, and cleanup to output the final wine recommendation datafram
  """
  congruent = congruent.head(max(2, 4-len(contrasting)))
  contrasting = contrasting.head(max(2, 4-len(congruent)))
  merged_df = pd.concat([congruent, contrasting], axis=0, ignore_index=True)
  merged_df.drop(columns=['aroma vector', 'weight', 'sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter'], inplace=True)
  return merged_df



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
    wine_vector_df[taste] = wine_vector_df[col_name].map(lambda scalar: standardize_wine_nonaroma_scalar(taste, scalar))

  # get all key attributes for a list of food, including average food vector, and nonaroma scaled values
  print('\nPlease input a list of food ingredients, separated by comma:')
  food_list = [f.strip(' ') for f in input().split(',') if f]
  if not food_list: print('No food ingredients provided!\n'); quit()
  food_nonaroma_values, food_weight, food_avg_vector = retrieve_all_food_attributes(food_list, food_nonaroma_df, core_nonaromas, food_tokenizer, food_phraser, aroma_descriptor_mapper, word2vec)

  # filter wine dataframe by a list of country
  print('\nPlease input a list of countries for wine selection, separated by comma:')
  geo_picker = [c.strip(' ') for c in input().split(',') if c]
  wine_vector_df = pick_wines_by_geo(wine_vector_df, geo_picker)

  # now, utilizing the defined pairing rules, 4 wines are picked consisting of congruent and/or contrasting pairing, as well as the most impactful descriptors of each wine
  wine_recommendations = nonaroma_ruling(wine_vector_df, food_nonaroma_values, food_weight)
  wine_recommendations = congruent_or_contrasting_pairing(wine_recommendations, food_nonaroma_values)
  wine_recommendations = sort_by_aroma_similarity(wine_recommendations, food_avg_vector)
  wine_recommendations['most_impactful_descriptor'] = wine_recommendations.apply(lambda w: get_most_impactful_descriptors(wine_descriptor_df, w['Variety'], w['Geo'], food_avg_vector, word2vec), axis=1)

  congruent_recommendations = wine_recommendations.loc[wine_recommendations['pairing_type'] == 'congruent'].head(4).reset_index(drop=True)
  contrasting_recommendations = wine_recommendations.loc[wine_recommendations['pairing_type'] == 'contrasting'].head(4).reset_index(drop=True)

  final_wine_recommendations = merge_congruent_and_contrasting(congruent_recommendations, contrasting_recommendations)

  # make visualization plots

  wine_plot_data = [dict(final_wine_recommendations.iloc[w]) for w in range(len(final_wine_recommendations))]
  food_plot_data = {taste + ' scalar': food_nonaroma_values[taste][0] for taste in food_nonaroma_values}
  food_plot_data['weight scalar'] = food_weight[0]
  food_plot_data['food'] = ', '.join(food_list)
  plot_wine_recommendations(wine_plot_data, food_plot_data)