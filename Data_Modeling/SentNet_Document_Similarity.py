# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 01:10:33 2018

@author: GTayl
"""

# SenNet Document Similarity
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

# Import the required packages
import gensim
import os
import collections
import smart_open
import random


# =============================================================================
# Part #2 - Document Similarity  
# =============================================================================


# Define the function to tokenize and tag documents required for Gensim
def read_corpus(data, field, target, tokens_only=False):
    for index, row in data.iterrows():
        if tokens_only:
            #yield gensim.utils.simple_preprocess(row[field])
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]))
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]), str(row[target]))

train_corpus = list(read_corpus(train, 'essay', target))
test_corpus = list(read_corpus(test, 'essay', target))

# Defining the function to train and return the Gensim Model
def Doc2Vec_Training_Model(train_corpus, field, target):
    
    # Preprocess text for Doc2Vec Model
    
    # Train the Model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=6, epochs=100)
    model.build_vocab(train_corpus)
    %time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    return(model)

# Obtain Estimates
gensim_training = Doc2Vec_Training_Model(train_corpus, 'essay', target)
    

def Doc2Vec_Training_Features(model, train_corpus, scores_join, target):
    
    # Initialize the Dictionary to Return
    estimates = pd.DataFrame(columns=['Doc_ID','Pred_Class','Pred_Prob'])

    # Establish a For loop to loop over all documents
    for doc_id in range(len(train_corpus)):
        
        # Calculate Document Similarity Measures
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        sims = pd.DataFrame(sims, columns=['Pred_Class','Prob'])
        pred_class = sims['Pred_Class'].iloc[sims['Prob'].argmax()]
        pred_prob = sims['Prob'].iloc[sims['Prob'].argmax()]
        
        '''
        # Merge Similarity Measures with Original Scores
        sims = pd.merge(sims, scores_join, on=['Pred_Class_Index','Pred_Class_Index'])
        
        # Obtain the score of the most similar document, save as the predicted class
        pred_class = sims[target].iloc[sims['Prob'].argmax()]
        
        # Multiply the probability of a document match by the score of that document to get the average weighted class
        sims['Weighted_Class'] = sims.apply(lambda row: (row[target]*row['Prob']), axis=1)
        weighted_class = sims['Weighted_Class'].mean()
        '''
        # Append the estimates to the estimates dataframe
        estimates = estimates.append({'Doc_ID':doc_id, 'Pred_Class':pred_class, 'Pred_Prob':pred_prob}, ignore_index=True)
            
    return(estimates)

%time train_esimates = Doc2Vec_Training_Features(gensim_training, train_corpus, scores_join, target)

pred_crosstab = pd.crosstab(train_esimates["Pred_Class"],train[target])
print(pred_crosstab)

%time test_esimates = Doc2Vec_Training_Features(gensim_training, test_corpus, scores_join, target)
pred_crosstab = pd.crosstab(test_esimates["Pred_Class"],test[target])
print(pred_crosstab)

# calculate model accuracy
preds = pd.DataFrame(test_esimates)
preds.reset_index(drop=True, inplace=True)
target_test = pd.DataFrame(test[target])
target_test.reset_index(drop=True, inplace=True)

accuracy = pd.concat([target_test, preds],axis=1)
accuracy['correct_pred'] = accuracy.apply(lambda row: abs(int(row[target])-int(row["Pred_Class"])), axis=1)
model_error = sum(accuracy['correct_pred'])

# overall accuracy rate
accurate_count = 0
for i in range(0,len(accuracy)):
    try:
        if accuracy['correct_pred'].iloc[i]==0:
            accurate_count += 1
    except:
        pass

total_count = pred_crosstab.sum().sum()
accuracy_rate = accurate_count/total_count
print("Total Accuracy Rate: "+str(accuracy_rate))

# average standard error
std_error = model_error/total_count
print("Model Standard Error: "+str(std_error))