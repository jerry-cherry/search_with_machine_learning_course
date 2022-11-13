import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
stemmed_file_name = r'/workspace/datasets/fasttext/stemmed_queries.txt'
count_file_name = r'/workspace/datasets/fasttext/count_queries.txt'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
def normalize_query(query):
    normalized = " ".join(stemmer.stem(token) for token in re.sub('[^a-zA-Z0-9]', ' ', query.lower()).split())
    return normalized if normalized else ' '

if os.path.exists(stemmed_file_name):
    print("Using the stemmed file: %s" % stemmed_file_name)
    queries_df = pd.read_csv(stemmed_file_name)[['category', 'query']]
else:
    print("Normalizing queries in %s" % queries_file_name)
    queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
    queries_df = queries_df[queries_df['category'].isin(categories)]

    # Convert queries to lowercase, and apply stemming.
    queries_df['query'] = queries_df['query'].apply(normalize_query)
    queries_df.to_csv(stemmed_file_name, header=True, escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

# Roll up categories to ancestors to satisfy the minimum number of queries per category.
hasRollupCategories = True;
while hasRollupCategories:
    countsByCategory = queries_df.groupby('category').count()
    print("%d categories are available" % countsByCategory.shape[0])
    countsByCategory.to_csv(count_file_name, header=True, escapechar='\\', quoting=csv.QUOTE_NONE, index=True)
    countsByCategory = countsByCategory[countsByCategory['query'] < min_queries]
    if countsByCategory.shape[0] > 0:
        print("%d cagetories are rolling up" % countsByCategory.shape[0])
        rollupCategories = countsByCategory.join(parents_df.set_index('category'), how='inner')
        for index, row in rollupCategories.iterrows():
            if row['parent']:
                queries_df.loc[queries_df['category'] == index, 'category'] = row['parent']
    else:
        hasRollupCategories = False

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
