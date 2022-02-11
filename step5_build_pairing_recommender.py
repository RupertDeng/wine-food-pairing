from data_importer import import_wine_variety_vector_info, import_wine_variety_descriptor_info, import_food_nonaroma_info, import_word2vec_model




if __name__ == '__main__':

  # import wine and food processed data, cell with vector data has already been converted to numpy ndarray of float64.
  wine_vector_df = import_wine_variety_vector_info()
  wine_descriptor_df = import_wine_variety_descriptor_info()
  food_nonaroma_df = import_food_nonaroma_info()

  # for the nonaroma scalars, sweet/salt/piquant/fat are in order, weight/acid/bitter need to be flipped to match common sense (larger value means more)
  for taste in ['weight', 'acid', 'bitter']:
    col_name = taste + ' scalar'
    wine_vector_df[col_name] = wine_vector_df[col_name].map(lambda x: -x)

  
  
  