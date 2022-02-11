from data_importer import import_wine_variety_vector_info, import_wine_variety_descriptor_info, import_food_nonaroma_info, import_word2vec_model

def retrieve_all_food_attributes(food_list):
  pass





if __name__ == '__main__':

  # import wine and food processed data, cell with vector data has already been converted to numpy ndarray of float64.
  wine_vector_df = import_wine_variety_vector_info()
  wine_descriptor_df = import_wine_variety_descriptor_info()
  food_nonaroma_df = import_food_nonaroma_info()



  
  
  