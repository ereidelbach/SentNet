# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:56:51 2018

@author: GTayl
"""

###############################################################################
# Package Import
###############################################################################

# Read in the required packages
import math
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# import functions from other python files
from Data_Ingest.SentNet_Data_Prep_Functions import Ingest_Training_Data

#from SentNet_Data_Feature_Extraction_V3 import Readability_Features

from Data_Preprocessing.SentNet_Data_Feature_Extraction import Word_Features, \
Word_Edge_Features, Word_Centrality_Features, Synset_Features, \
Synset_Edge_Features, Synset_Centrality_Features, Readability_Features

from Data_Modeling.Term_Clustering import Clustering_Features, \
Cluster_Concentrations, Synset_Clustering_Features, Synset_Concentrations

from Data_Modeling.SentNet_Document_Matching import \
Document_Matching_Training, Document_Matching_Testing

from Data_Modeling.SentNet_Document_Similarity import \
Document_Similarity_Training, Document_Similarity_Testing

###############################################################################
# Data Ingest 
###############################################################################

# Establish the project root directory
path_project_root = Path('SentNet_Training_Master.py').resolve().parent

################################## .Docx Data Ingest ##########################
# Assumes the Set1 Data is in the path: SentNet\Data\Set1\docx\Images
path_data = Path(path_project_root, 'Data','Set1','docx','Images')
# It will place your image files in: SentNet\Data\Set1\docx\Images\Unpacked
path_images = Path(path_project_root,'Data','Set1','docx','Images', 'Unpacked')
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
data = pd.DataFrame(temp_list)

'''
############################### Spreadsheet Data Ingest #######################
# Alternatively read in data from a pre-populated spreadsheet (or database table)
# This assumes we wish to read the csv located in the path: 
#      SentNet\Data\Set1
data_train = Path('Data','Set1','docx','Images')
data = pd.DataFrame.from_csv(data_train, sep='\t', header=0, encoding='ISO-8859-1')

# Select any subset (Scorecard) that you want to use for training
data = data[data['essay_set']==1]
print("Done Data Ingest")
'''

###############################################################################
# Train and Test Split
###############################################################################

train, test = train_test_split(data, test_size=0.2)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

###############################################################################
# Feature Extraction 
###############################################################################

# Define the minimum threshold (the number of documents a feature must appear 
#   in) for a feature to be included in our analysis
limit = round(0.02*len(train))
target = 'domain1_score'
doc = 'essay'

# Calculate readability features for each document
readability_features = Readability_Features(train, doc)

# Matching (most similar) document score (Doc2Vec)
matching_docs = Document_Matching_Training(train, doc, target, limit=0)
# Saving this model to apply to testing/scoring data later on
gensim_model_matching = matching_docs['gensim_model'] 
# Saving this as an input to testing/scoring data later on 
scores_join = matching_docs['scores_join'] 
matching_docs = matching_docs['train_estimates']

# General Similarity score (Doc2Vec)
similar_docs = Document_Similarity_Training(train, doc, target)
gensim_model_similarity = similar_docs['gensim_training_similiarity']
similar_docs = similar_docs['train_esimates']

# Calculate word level features for each document
word_features = Word_Features(train, doc, limit)
selected_words = list(word_features['selected_words'])
word_features = word_features['word_matrix_features']

# Calculate word graph features for each document
word_edge_features = Word_Edge_Features(train, doc, limit)
word_edge_list = word_edge_features['master_edge_list']
word_edge_features = word_edge_features['edges_matrix_features']
word_centrality_features = Word_Centrality_Features(train, doc, selected_words)
word_clusters = Clustering_Features(word_edge_list, limit)
word_cluster_features = Cluster_Concentrations(train, doc, word_clusters)

# Calculate synset level features for each document
synset_features = Synset_Features(train, doc, limit)
selected_synsets = synset_features['selected_synsets']
synset_features = synset_features['synset_matrix_features']

# Calculate synset graph features for each document
synset_edge_features = Synset_Edge_Features(train, doc, limit)
synset_edge_list = synset_edge_features['master_edge_list_synset']
synset_edge_features = synset_edge_features['edges_matrix_features_synset']
synset_centrality_features = Synset_Centrality_Features(
        train, doc, selected_synsets)
synset_clusters = Synset_Clustering_Features(synset_edge_list, limit)
synset_cluster_features = Synset_Concentrations(train, doc, synset_clusters)

###############################################################################
# Modeling 
###############################################################################

# Merge feature space to create training dataset
train_data = pd.concat([readability_features, 
                        matching_docs['Matching_Pred_Class'], 
                        similar_docs['Sim_Pred_Class'],
                        word_features,
                        word_edge_features,
                        word_centrality_features,
                        word_cluster_features,
                        synset_features,
                        synset_edge_features,
                        synset_centrality_features,
                        synset_cluster_features
                        ]
                        ,axis=1)

train_data = train_data.replace('N/A',0)
train_data = train_data.replace(np.nan,0)
train_data = train_data.loc[:,~train_data.columns.duplicated()]

# Train inital random forest for feature selection
rfc = RandomForestClassifier(n_estimators=10000, n_jobs=7)
rfc.fit(train_data,train[target])

# estimate the feature importance
feature_importance = pd.DataFrame(
        rfc.feature_importances_,columns=['importance'])
feature_importance = pd.concat([pd.DataFrame(
        list(train_data.columns)),feature_importance],axis=1)
modeling_features = list(
        feature_importance[feature_importance['importance']>=0.0001][0])

# train scoring model
rfc = RandomForestClassifier(
        n_estimators=2000, n_jobs=7, max_depth=8, min_samples_leaf=3)
rfc.fit(train_data[modeling_features],train[target])
feature_importance2 = pd.DataFrame(rfc.feature_importances_,columns=['importance'])
feature_importance2 = pd.concat([pd.DataFrame(list(
        train_data[modeling_features].columns)),feature_importance2],axis=1)

# calculate predicted values for training set
preds = rfc.predict(train_data[modeling_features])
pred_crosstab = pd.crosstab(train[target],preds)
print("TRAINING")
print(pred_crosstab)
print(" ")

###############################################################################
# Scoring 
###############################################################################

# Calculate readability features for each document
readability_features_score = Readability_Features(test, doc)

# Matching (most similar) document score (Doc2Vec)
matching_docs_score = Document_Matching_Testing(
        test, doc, target, scores_join, gensim_model_matching, limit=0)

# General Similarity score (Doc2Vec)
similar_docs_score = Document_Similarity_Testing(
        train, doc, target, gensim_model_similarity, limit=0)

# Calculate word level features for each document
word_features_score = Word_Features(
        test, doc, limit=0)['word_matrix_features']
word_features_score = word_features_score[
        word_features_score.columns.intersection(selected_words)]

# Calculate word graph features for each document
word_edge_features_score = Word_Edge_Features(
        test, doc, limit)['edges_matrix_features']
word_edge_features_score = word_edge_features_score[
        word_edge_features_score.columns.intersection(
                list(set(word_edge_list['edge_id'])))]
word_centrality_features_score = Word_Centrality_Features(
        test, doc, selected_words)
word_cluster_features_score = Cluster_Concentrations(test, doc, word_clusters)

# Calculate synset level features for each document
synset_features_score = Synset_Features(
        test, doc, limit=0)['synset_matrix_features']
synset_features_score = synset_features_score[
        synset_features_score.columns.intersection(selected_synsets)]

# Calculate synset graph features for each document
synset_edge_features_score = Synset_Edge_Features(
        test, doc, limit)['edges_matrix_features_synset']
synset_edge_features_score = synset_edge_features_score[
        synset_edge_features_score.columns.intersection(
                list(set(synset_edge_list['edge_id'])))]
synset_centrality_features_score = Synset_Centrality_Features(
        test, doc, selected_synsets)
synset_cluster_features_score = Synset_Concentrations(test, doc, synset_clusters)

# Preparing dataset for predictions
test_data = pd.concat([readability_features_score, 
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

test_data = test_data.replace('N/A',0)
test_data = test_data.replace(np.nan,0)
test_data = test_data.loc[:,~test_data.columns.duplicated()]

train_test_diff = list(set(modeling_features)-set(test_data.columns))

for i in train_test_diff:
    test_data[i]=0

# test scoring model
preds = rfc.predict(test_data[modeling_features])
pred_crosstab = pd.crosstab(test[target],preds)
print("TESTING")
print(pred_crosstab)
print(" ")

accurate_count = 0
for i in range(0,len(pred_crosstab-1)):
    try:
        accurate_count += pred_crosstab.iloc[i,i]
    except:
        pass

total_count = pred_crosstab.sum().sum()
accuracy_rate = accurate_count/total_count
print("Total Accuracy Rate: "+str(accuracy_rate))

accuracy = pd.concat([pd.DataFrame(test[target]), pd.DataFrame(preds)],axis=1)
accuracy['correct_pred'] = accuracy.apply(
        lambda row: math.sqrt((row[target]-row[0])**2), axis=1)
accuracy['correct_buffer'] = accuracy['correct_pred']<=1
model_error = accuracy['correct_pred'].mean()
print("Model Error: "+str(model_error))

adjusted_accuracy = (accuracy['correct_buffer']==1).sum()/len(test)
print("Adjusted Accuracy: "+str(adjusted_accuracy))

model_results = {'model_error':model_error,
                 'accuracy_rate':accuracy_rate,
                 'adjusted_accuracy':adjusted_accuracy,
                 'pred_crosstab':pred_crosstab,
                 'feature_importance':feature_importance
                 }