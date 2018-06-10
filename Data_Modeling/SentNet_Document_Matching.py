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
# Part #1 - Document Matching  
# =============================================================================

# Select the target that you want to estimate
target = 'rater1_domain1'

'''
# Define the function to tokenize and tag documents required for Gensim
def read_corpus(data, field, target, tokens_only=False):
    for index, row in data.iterrows():
        if tokens_only:
            #yield gensim.utils.simple_preprocess(row[field])
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]))
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]), str(row[target]))
'''
            
def read_corpus(data, field, target, tokens_only=False):
    for index, row in data.iterrows():
        if tokens_only:
            #yield gensim.utils.simple_preprocess(row[field])
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]))
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]), [index])

# Develop the training corpus
train_corpus = list(read_corpus(train, 'essay', target))

# Defining the function to train and return the Gensim Model
def Doc2Vec_Training_Model(train_corpus, field, target):
    
    # Preprocess text for Doc2Vec Model
    
    # Train the Model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=1, epochs=10)
    model.build_vocab(train_corpus)
    %time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    return(model)

# Obtain Estimates
gensim_training = Doc2Vec_Training_Model(train_corpus, 'essay', target)

scores_join = pd.DataFrame(train[target])
scores_join['Pred_Class_Index'] = scores_join.index

def Doc2Vec_Training_Features(model, train_corpus, scores_join, target):
    
    # Initialize the Dictionary to Return
    estimates = pd.DataFrame(columns=['Doc_ID','Pred_Class','Pred_Prob','Weighted_Class'])

    # Establish a For loop to loop over all documents
    for doc_id in range(len(train_corpus)):
        
        # Calculate Document Similarity Measures
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        sims = pd.DataFrame(sims, columns=['Pred_Class_Index','Prob'])
        # Merge Similarity Measures with Original Scores
        sims = pd.merge(sims, scores_join, on=['Pred_Class_Index','Pred_Class_Index'])
        
        # Obtain the score of the most similar document, save as the predicted class
        pred_class = sims[target].iloc[sims['Prob'].argmax()]
        pred_prob = sims['Prob'].iloc[sims['Prob'].argmax()]
        
        # Multiply the probability of a document match by the score of that document to get the average weighted class
        sims['Weighted_Class'] = sims.apply(lambda row: (row[target]*row['Prob']), axis=1)
        weighted_class = sims['Weighted_Class'].mean()
        
        # Append the estimates to the estimates dataframe
        estimates = estimates.append({'Doc_ID':doc_id, 'Pred_Class':pred_class, 'Pred_Prob':pred_prob, 'Weighted_Class':weighted_class}, ignore_index=True)
            
    return(estimates)
        
%time train_esimates = Doc2Vec_Training_Features(gensim_training, train_corpus, scores_join, target)

pred_crosstab = pd.crosstab(train_esimates["Pred_Class"],train[target])
print(pred_crosstab)

# Testing
test_corpus = list(read_corpus(test, 'essay', target))

%time test_estimates = Doc2Vec_Training_Features(gensim_training, test_corpus, scores_join, target)

pred_crosstab = pd.crosstab(test_esimates["Pred_Class"],test[target])
print(pred_crosstab)
