# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:56:51 2018

@author: GTayl
"""

# Read in the required packages

from SentNet_Data_Feature_Extraction_V3.py import Readability_Features

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
# Feature Extraction 
###############################################################################################

# Define the minimum threshold (the number of documents a feature must appear in) for a feature to be included in our analysis
limit = round(0.01*len(data))

# Calculate readability features for each document
readability_features = Readability_Features(data, 'essay')

# Calculate word level features for each document
word_features = Word_Features(data, target, limit)
selected_words = word_features['selected_words']
word_features = word_features['word_matrix_features']

# Calculate word graph features for each document
word_edge_features = Word_Edge_Features(data, 'essay', limit)
word_centrality_features = Word_Centrality_Features(data, target, selected_words)

# Calculate synset level features for each document
synset_features = Synset_Features(data, 'essay', limit)
selected_synsets = synset_features['selected_synsets']
synset_features = synset_features['synset_matrix_features']

# Calculate synset graph features for each document
synset_edge_features = Synset_Edge_Features(data, 'essay', limit)
synset_centrality_features = Synset_Centrality_Features(data, target, selected_synsets)


