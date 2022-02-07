from data_importer import import_processed_wine_variety_data


# for the nonaroma scalars, sweet/salt/piquant/fat are in order, weight/acid/bitter need to be flipped

wine_variety_df = import_processed_wine_variety_data()
print(wine_variety_df['aroma descriptors'])