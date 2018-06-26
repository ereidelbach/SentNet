#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
:DESCRIPTION:
    While it would be beneficial to frequently retrain the random forest models 
    as new scored documents become available, we forsee the need to score 
    additional documents without retraining the SentNet models first. 
    
    When this is the case, it would be more effective to import the already 
    trained models (if your previous sesssion ended or shutdown) and simply 
    run new observations through those models(as opposed to having to retrain 
    the models from scratch). When this is the case we can use the script below 
    to score new observations/documents using an existing model.

:REQUIRES:
    Note: This requires that you outpt the final model and feature set as part 
    of the SentNet_Training_Master.py script. 
    
:TODO:
    NONE
"""
#==============================================================================
# Package Import
#==============================================================================
import datetime
import numpy as np
from pathlib import Path
import pandas as pd
import pickle

# import functions from other python files
from Data_Ingest.SentNet_Data_Prep_Functions import Ingest_Training_Data

from Data_Preprocessing.SentNet_Data_Feature_Extraction import Word_Features, \
Word_Edge_Features, Word_Centrality_Features, Synset_Features, \
Synset_Edge_Features, Synset_Centrality_Features, Readability_Features

from Data_Modeling.Term_Clustering import Cluster_Concentrations, Synset_Concentrations

from Data_Modeling.SentNet_Document_Matching import Document_Matching_Testing

from Data_Modeling.SentNet_Document_Similarity import Document_Similarity_Testing

#==============================================================================
# Data Ingest 
#==============================================================================

# Establish the project root directory
path_project_root = Path('SentNet_Training_Master.py').resolve().parent

# Specify the data folder that we want to model
data_set = 'Set1'

################################## .Docx Data Ingest ##########################
# This assumes the Scoring Data is in the path: SentNet\Data\Set1\docx\Images
path_data = Path('Data','Set1','docx','Images')
# It will place your image files in: SentNet\Data\Set1\docx\Images\Unpacked
path_images = Path('Data','Set1','docx','Images', 'Unpacked')
# Ingest the training data
data_raw = Ingest_Training_Data(path_data, path_images)

temp_list = []
for index, row in data_raw.iterrows():
    # luckily, everything is broken up by line splits so let's break up the 
    #   text that way
    doc_elements = row['Doc_Text'].split('\n')
    # delete empty elements
    doc_elements = [x for x in doc_elements if x != '']
    # remove any rows that are similar to 'Image 1', 'Image 2', etc...
    doc_elements = [x for x in doc_elements if 'Image ' not in x]
    # create a dictionary where the key is the variable name and the value is 
    #   score/text from the document
    doc_dict = {}
    for (key, value) in zip(doc_elements[::2], doc_elements[1::2]):
        doc_dict[key] = value
    # throw in the other row info into the dict
    doc_dict['Doc_Title'] = row['Doc_Title']
    doc_dict['Doc_Images'] = row['Doc_Images']
    temp_list.append(doc_dict)
   
# turn the list of dictionaries into a new dataframe with all the the info
#   necessary to proceed with analysis and modeling
score = pd.DataFrame(temp_list)

'''
############################### Spreadsheet Data Ingest #######################
# Alternatively read in data from a pre-populated spreadsheet (or database table)
# This assumes we wish to read the csv located in the path: 
#      SentNet\Data\Set1
data_train = Path('Data','Set1','docx','Images')
score = pd.DataFrame.from_csv(data_train, sep='\t', header=0, encoding='ISO-8859-1')

# Select any subset (Scorecard) that you want to use for training
score.reset_index(drop=True, inplace=True)
print("Done Data Ingest")
'''

#==============================================================================
# Model Ingest 
#==============================================================================
path_to_features = "INPUT PATH TO SAVED FEATURES HERE"
path_to_model = "INPUT PATH TO MODEL HERE"

# Read in Selected Feature Set
modeling_features = list(pd.read_csv(path_to_features))

# Read in Random Forest Model
with open('path/to/file', 'rb') as f:
    rfc = pickle.load(f)

#==============================================================================
# Model_Parameters
#==============================================================================
# Define the minimum threshold (the number of documents a feature must 
#   appear in) for a feature to be included in our analysis
limit = round(0.02*len(score))

# The element/scorecard you are attempting to estimate
target = 'domain1_score'

# The name of the column in the dataframe which contains the document text 
#   to be scored
doc = 'essay'

#==============================================================================
# Feature Extraction 
#==============================================================================
# Calculate readability features for each document
readability_features_score = Readability_Features(score, doc)

# Matching (most similar) document score (Doc2Vec)
matching_docs_score = Document_Matching_Testing(
        score, doc, target, scores_join, gensim_model_matching, limit=0)

# General Similarity score (Doc2Vec)
similar_docs_score = Document_Similarity_Testing(
        score, doc, target, gensim_model_similarity, limit=0)

# Calculate word level features for each document
word_features_score = Word_Features(score, doc, limit=0)['word_matrix_features']
word_features_score = word_features_score[
        word_features_score.columns.intersection(selected_words)]

# Calculate word graph features for each document
word_edge_features_score = Word_Edge_Features(
        score, doc, limit)['edges_matrix_features']
word_edge_features_score = word_edge_features_score[
        word_edge_features_score.columns.intersection(
                list(set(word_edge_list['edge_id'])))]
word_centrality_features_score = Word_Centrality_Features(
        score, doc, selected_words)
word_cluster_features_score = Cluster_Concentrations(score, doc, word_clusters)

# Calculate synset level features for each document
synset_features_score = Synset_Features(
        score, doc, limit=0)['synset_matrix_features']
synset_features_score = synset_features_score[
        synset_features_score.columns.intersection(selected_synsets)]

# Calculate synset graph features for each document
synset_edge_features_score = Synset_Edge_Features(
        score, doc, limit)['edges_matrix_features_synset']
synset_edge_features_score = synset_edge_features_score[
        synset_edge_features_score.columns.intersection(
                list(set(synset_edge_list['edge_id'])))]
synset_centrality_features_score = Synset_Centrality_Features(
        score, doc, selected_synsets)
synset_cluster_features_score = Synset_Concentrations(
        score, doc, synset_clusters)

# Preparing dataset for predictions
score_data = pd.concat([readability_features_score, 
                        matching_docs_score['Matching_Pred_Class'], 
                        similar_docs_score['Sim_Pred_Class'],
                        word_features_score,
                        word_edge_features_score,
                        word_centrality_features_score,
                        word_cluster_features_score,
                        synset_features_score,
                        synset_edge_features_score,
                        synset_centrality_features_score,
                        synset_cluster_features_score
                        ]
                        ,axis=1)

score_data = score_data.replace('N/A',0)
score_data = score_data.replace(np.nan,0)
score_data = score_data.loc[:,~score_data.columns.duplicated()]

train_test_diff = list(set(modeling_features)-set(score_data.columns))

for i in train_test_diff:
    score_data[i]=0

#==============================================================================
# Model Scoring
#==============================================================================    
preds = rfc.predict(score_data[modeling_features])

#==============================================================================
# Saving Results 
#==============================================================================
'''
In a production system, depending on your exisisting data management systems/
processes you could dump these results to a database or other location. As
a temporary standin for a finalized solution, we write our results to an
excel worksheet.
'''
date = str(datetime.date.today())
file_name = ("SentNet_Scoring_Predictions_"+date+".xlsx")
preds.to_excel(file_name)
