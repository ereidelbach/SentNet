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
# Document Similarity  
# =============================================================================


# Define the function to tokenize and tag documents required for Gensim
def read_corpus(data, field, target, tokens_only=False):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    for index, row in data.iterrows():
        if tokens_only:
            #yield gensim.utils.simple_preprocess(row[field])
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]))
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]), str(row[target]))


# Defining the function to train and return the Gensim Model
def Doc2Vec_Training_Model(train_corpus):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Preprocess text for Doc2Vec Model
    
    # Train the Model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=6, epochs=100)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    return(model)
    

def Doc2Vec_Sim_Estimates(model, train_corpus):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
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


def Document_Similarity_Training(train, doc, target):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Develop Training Corpus
    train_corpus = list(read_corpus(train, doc, target))
    
    # Define Model
    gensim_training_similiarity = Doc2Vec_Training_Model(train_corpus)
    
    # Obtain estimated document similarities
    train_esimates = Doc2Vec_Sim_Estimates(gensim_training_similiarity, train_corpus)
    
    return({'train_esimates':train_esimates, 'gensim_training_similiarity':gensim_training_similiarity})


def Document_Similarity_Testing(train, doc, target, gensim_model):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Develop Training Corpus
    test_corpus = list(read_corpus(test, 'essay', target))
    
    # Obtain estimated document similarities
    test_esimates = Doc2Vec_Sim_Estimates(gensim_model, test_corpus)
    
    return(train_esimates)


'''