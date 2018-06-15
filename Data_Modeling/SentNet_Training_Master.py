# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:56:51 2018

@author: GTayl
"""

# Read in the required packages

from SentNet_Data_Feature_Extraction_V3.py import Readability_Features
from sklearn.model_selection import train_test_split

###############################################################################################
# Data Ingest 
###############################################################################################

################################## .Docx Data Ingest ##########################################
'''
# Specify a folder that contains the training document in .docx format
doc_folder_path = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\Example_Docs"
img_dir = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Photos\\test_photos"

# Use the Ingest Training Data function from SentNet_Data_Prep_Functions.py to read in the training data 
data = Ingest_Training_Data(doc_folder_path, img_dir)
'''
############################### Spreadsheet Data Ingest #######################################
# Alternatively read in data from a prepoulated spreadsheet (or database table)
traing_data = 'C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\training_set_rel3.tsv'
data = pd.DataFrame.from_csv(traing_data, sep='\t', header=0, encoding='ISO-8859-1')

# Select any subset (Scorecard) that you want to use for training
data = data[data['essay_set']==1]
print("Done Data Injest")
time1 = time.time()
print(time1-start)

###############################################################################################
# Train and Test Split
###############################################################################################

train, test = train_test_split(data, test_size=0.2)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

###############################################################################################
# Feature Extraction 
###############################################################################################

# Define the minimum threshold (the number of documents a feature must appear in) for a feature to be included in our analysis
limit = round(0.01*len(train))

# Calculate readability features for each document
readability_features = Readability_Features(train, 'essay')

# Matching (most similar) document score (Doc2Vec)
target = 'rater1_domain1'
matching_docs = Document_Matching_Training(train, 'essay', target, limit=0)
gensim_model_matching = matching_docs['gensim_model'] # Saving this model to apply to testing/scoring data later on
scores_join = matching_docs['scores_join'] # Saving this as an input to testing/scoring data later on 
matching_docs = matching_docs['train_estimates']

# General Similarity score (Doc2Vec)
similar_docs = Document_Similarity_Training(train, 'essay', target)
gensim_model_similarity = similar_docs['gensim_training_similiarity']
similar_docs = similar_docs['train_esimates']

# Calculate word level features for each document
word_features = Word_Features(train, target, limit)
selected_words = word_features['selected_words']
word_features = word_features['word_matrix_features']

# Calculate word graph features for each document
word_edge_features = Word_Edge_Features(train, 'essay', limit)
word_edge_list = word_edge_features['master_edge_list']
word_edge_features = word_edge_features['edges_matrix_features']
word_centrality_features = Word_Centrality_Features(train, target, selected_words)
word_clusters = Clustering_Features(word_edge_list, limit)
word_cluster_features = Cluster_Concentrations(train, word_clusters)

# Calculate synset level features for each document
synset_features = Synset_Features(train, 'essay', limit)
selected_synsets = synset_features['selected_synsets']
synset_features = synset_features['synset_matrix_features']

# Calculate synset graph features for each document
synset_edge_features = Synset_Edge_Features(train, 'essay', limit)
synset_edge_list = synset_edge_features['master_edge_list_synset']
synset_edge_features = synset_edge_features['edges_matrix_features_synset']
synset_centrality_features = Synset_Centrality_Features(train, target, selected_synsets)
# Synset Cluster Represnetations
# Synset Cluster Features

# Image similarity
