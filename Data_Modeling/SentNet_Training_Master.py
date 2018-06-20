# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:56:51 2018

@author: GTayl
"""

# Read in the required packages

from SentNet_Data_Feature_Extraction_V3 import Readability_Features
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
limit = round(0.02*len(train))
target = 'domain1_score'
doc = 'essay'

# Calculate readability features for each document
readability_features = Readability_Features(train, doc)

# Matching (most similar) document score (Doc2Vec)
matching_docs = Document_Matching_Training(train, doc, target, limit=0)
gensim_model_matching = matching_docs['gensim_model'] # Saving this model to apply to testing/scoring data later on
scores_join = matching_docs['scores_join'] # Saving this as an input to testing/scoring data later on 
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
synset_centrality_features = Synset_Centrality_Features(train, doc, selected_synsets)
synset_clusters = Synset_Clustering_Features(synset_edge_list, limit)
synset_cluster_features = Synset_Concentrations(train, doc, synset_clusters)

###############################################################################################
# Modeling 
###############################################################################################

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
                        synset_centrality_matrix,
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
feature_importance = pd.DataFrame(rfc.feature_importances_,columns=['importance'])
feature_importance = pd.concat([pd.DataFrame(list(train_data.columns)),feature_importance],axis=1)
modeling_features = list(feature_importance[feature_importance['importance']>=0.0001][0])

# train scoring model
rfc = RandomForestClassifier(n_estimators=2000, n_jobs=7, max_depth=8, min_samples_leaf=3)
rfc.fit(train_data[modeling_features],train[target])
feature_importance2 = pd.DataFrame(rfc.feature_importances_,columns=['importance'])
feature_importance2 = pd.concat([pd.DataFrame(list(train_data[modeling_features].columns)),feature_importance2],axis=1)

# calculate predicted values for training set
preds = rfc.predict(train_data[modeling_features])
pred_crosstab = pd.crosstab(train[target],preds)
print("TRAINING")
print(pred_crosstab)
print(" ")

################################################################################################
# Scoring 
###############################################################################################

# Calculate readability features for each document
readability_features_score = Readability_Features(test, doc)

# Matching (most similar) document score (Doc2Vec)
matching_docs_score = Document_Matching_Testing(test, doc, target, scores_join, gensim_model_matching, limit=0)

# General Similarity score (Doc2Vec)
similar_docs_score = Document_Similarity_Testing(train, doc, target, gensim_model_similarity, limit=0)

# Calculate word level features for each document
word_features_score = Word_Features(test, doc, limit=0)['word_matrix_features']
word_features_score = word_features_score[word_features_score.columns.intersection(selected_words)]

# Calculate word graph features for each document
word_edge_features_score = Word_Edge_Features(test, doc, limit)['edges_matrix_features']
word_edge_features_score = word_edge_features_score[word_edge_features_score.columns.intersection(list(set(word_edge_list['edge_id'])))]
word_centrality_features_score = Word_Centrality_Features(test, doc, selected_words)
word_cluster_features_score = Cluster_Concentrations(test, doc, word_clusters)

# Calculate synset level features for each document
synset_features_score = Synset_Features(test, doc, limit=0)['synset_matrix_features']
synset_features_score = synset_features_score[synset_features_score.columns.intersection(selected_synsets)]

# Calculate synset graph features for each document
synset_edge_features_score = Synset_Edge_Features(test, doc, limit)['edges_matrix_features_synset']
synset_edge_features_score = synset_edge_features_score[synset_edge_features_score.columns.intersection(list(set(synset_edge_list['edge_id'])))]
synset_centrality_features_score = Synset_Centrality_Features(test, doc, selected_synsets)
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
accuracy['correct_pred'] = accuracy.apply(lambda row: math.sqrt((row[target]-row[0])**2), axis=1)
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