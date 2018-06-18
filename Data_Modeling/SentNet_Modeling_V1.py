# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:13:57 2018

@author: GTayl
"""

###############################################################################################
################################# SentNet Modeling Process ####################################
###############################################################################################


###############################################################################################
############################### Import the required packages ##################################
###############################################################################################

# Import the required packages

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

###############################################################################################
################################### Import Data/Documents #####################################
###############################################################################################
'''
# Specify a folder that contains the training document in .docx format
doc_folder_path = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\Example_Docs"
img_dir = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Photos\\test_photos"

# Use the Ingest Training Data function from SentNet_Data_Prep_Functions.py to read in the training data 
data = Ingest_Training_Data(doc_folder_path, img_dir)

############################### Spreadsheet Data Ingest #######################################
# Alternatively read in data from a prepoulated spreadsheet (or database table)
traing_data = 'C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\training_set_rel3.tsv'
data = pd.DataFrame.from_csv(traing_data, sep='\t', header=0, encoding='ISO-8859-1')

# Select any subset (Scorecard) that you want to use for training
data = data[data['essay_set']==1]
'''
###############################################################################################
################################### Extract Feature Sets #####################################
###############################################################################################

# TBD


###############################################################################################
###################################### Train Models ###########################################
###############################################################################################

# Select target
target = 'rater1_domain1'
essay = pd.DataFrame(data['essay'])
essay.columns = ['raw_essay']
model_data = pd.concat([essay,doc_readability_features,synset_matrix_features,word_matrix_features,word_centrality_matrix,synset_centrality_matrix],axis=1)

# replace any N/A's
model_data = model_data.replace('N/A',0)
model_data = model_data.replace(np.nan,0)
model_data.reset_index(drop=True, inplace=True)

# select the targets
target_1 = data['rater1_domain1']
target_1.reset_index(drop=True, inplace=True)
target_2 = data['rater2_domain1']
target_2.reset_index(drop=True, inplace=True)
target_3 = data['domain1_score']
target_3.reset_index(drop=True, inplace=True)


# append the target data to the feature set
model_data = pd.concat([model_data, target_1, target_2, target_3],axis=1)

# select the train and test splits
train, test = train_test_split(model_data, test_size=0.2)

# obtain document similarity features



# select the model features
modeling_features = list(test.columns)
modeling_features.remove('rater1_domain1')
modeling_features.remove('rater2_domain1')
modeling_features.remove('domain1_score')
modeling_features.remove('ESSAY')

# train the model for feature selection
rfc = RandomForestClassifier(n_estimators=1000, n_jobs=7)
rfc.fit(train[modeling_features],train[target])

# estimate the feature importance
feature_importance = pd.DataFrame(rfc.feature_importances_,columns=['importance'])
feature_importance = pd.concat([pd.DataFrame(modeling_features),feature_importance],axis=1)
selected_features = list(feature_importance[feature_importance['importance']>=0.001][0])
modeling_features = selected_features

# train scoring model
rfc = RandomForestClassifier(n_estimators=100, n_jobs=7)
rfc.fit(train[modeling_features],train[target])

# calculate predicted values for training set
preds = rfc.predict(train[modeling_features])
pred_crosstab = pd.crosstab(train[target],preds)
print("TRAINING")
print(pred_crosstab)
print(" ")

# calculate predicted values for test set
preds = rfc.predict(test[modeling_features])
pred_crosstab = pd.crosstab(test[target],preds)
print("TESTING")
print(pred_crosstab)

# find the most common value/level for test set
naive = train[target].groupby(train[target]).count()
naive = naive.idxmax()

# calculate model accuracy
preds = pd.DataFrame(preds)
preds.reset_index(drop=True, inplace=True)
target_test = pd.DataFrame(test[target])
target_test.reset_index(drop=True, inplace=True)

accuracy = pd.concat([target_test, preds],axis=1)
accuracy['correct_pred'] = accuracy.apply(lambda row: abs(row[target]-row[0]), axis=1)
model_error = sum(accuracy['correct_pred'])

# calculate naive model error
accuracy['naive_pred'] = accuracy.apply(lambda row: abs(naive-row[0]), axis=1)
naive_error = sum(accuracy['naive_pred'])

# overall accuracy rate
accurate_count = 0
for i in range(0,len(pred_crosstab-1)):
    try:
        accurate_count += pred_crosstab.iloc[i,i]
    except:
        pass

total_count = pred_crosstab.sum().sum()
accuracy_rate = accurate_count/total_count
print("Total Accuracy Rate: "+str(accuracy_rate))

# relative standard error (model vs naive)
rel_accuracy_rate = (naive_error-model_error)/naive_error
print("Relative Accuracy Rate: "+str(rel_accuracy_rate))

# average standard error
std_error = model_error/total_count
print("Model Standard Error: "+str(std_error))

###############################################################################################
################################ Image Comparison & Scoring ###################################
###############################################################################################