# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 01:10:33 2018

@author: GTayl
"""

# =============================================================================
# Overview
# =============================================================================
'''
In this file we define the functions that are required to perform document matching with SentNet using a Doc2Vec model implimented in Gensim.
'''

# =============================================================================
# Document Matching Set-up
# =============================================================================

# Import the required packages
import gensim
import os
import collections
import smart_open
import random

# =============================================================================
# Document Matching Functions
# =============================================================================

# Defining Functions for Document Matching            
def read_corpus(data, field, tokens_only=False):
    '''
    Purpose: This function prepares text for processing within the Gensim model by converting documents to the doc2vec.TaggedDocument class format. This involves:
        
                1) Converting the provided text into a list of words used within the target document
                2) Storing a list tags associated with the list of terms. In this case we store the index of the essay associated with this word list
    
            This, in turn allows these documents to be scored and analyzed using the Doc2Vec model.
        
    Input: The following inputs are required for this function:
        
            1) data = a pandas DataFrame that contains the documents of interest
            2) field = the column/feature within the dataframe that contains the text to be analyzed
            3) tokens_only = If you would only like tokens produced for the document (word list), but don't have any tags for those documents (as with testing data) specify tokens_only=True
        
    Output: This function returns a doc2vec.TaggedDocument as decribed above.
        
    '''
    
    for index, row in data.iterrows():
        if tokens_only==True:
            #yield gensim.utils.simple_preprocess(row[field])
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]))
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]), [index])


# Defining the function to train and return the Gensim Model
def Doc2Vec_Training_Model(train_corpus, field):
    '''
    Purpose: This function uses a tagged document to generate a Doc2Vec model that can be used to identify similar/matching documents.
        
    Input: This function requires the following inputs:
        
            1) train_corpus = a doc2vec.TaggedDocument list (returned by the read_corpus function)
            2) field = the column/feature within the dataframe that contains the text to be analyzed
        
    Output: This function returns a gensim.models.doc2vec.Doc2Vec object that can be used to score documents later on
    '''
    
    # Preprocess text for Doc2Vec Model
    
    # Train the Model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=1, epochs=1000)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    return(model)
  
    
# Defining the function to return the estimated probabilities for each document
def Doc2Vec_Estimates_Training(model, train_corpus, scores_join, target, limit=0):
    '''
    Purpose: This function produces an estimated score for every document by attempting to identify the closest related/matching document to the target document.
             This requires the following steps:
                 
                 1) Initalize an empty estimates dataframe to hold estimated values in
                 2) For every document in the dataset find the second most similar document within the corpus to the document provided (the second most similar document is used, otherwise the document would match to itself)
                 3) Assign the value of the second most similar document to this document as well as the "probabiliy"/"similarity" of the match
                 4) If the probabily falls below a given threshold (usually .7) assign the predicted class to 0
                 5) Append the result to the estimates dataframe
        
    Input: This function requires the following inputs:
        
            1) model = a gensim.models.doc2vec.Doc2Vec object (can be generated using the Doc2Vec_Training_Model function)
            2) train_corpus = a doc2vec.TaggedDocument list (returned by the read_corpus function)
            3) scores_join = a Pandas dataframe that contains the index of all documents within the training set as well as the target score for those documents
            4) target = the target/feature you are attmeping to model/estiamte
            5) limit = the probability limit that a prediction must exceed for that prediction to be assigned/passed on (usually .7, but defaults to 0 if not assigned)
    
    Output: This function returns a pandas DataFrame with the following features:
        
            1) Doc_Id = The index of provided document
            2) Pred_Class = The predicted class/value of that document (assigned using the method described above)
            3) Pred_Prob = The Doc2Vec provided probability of a match
        
    '''
    
    # Initialize the Dictionary to Return
    estimates = pd.DataFrame(columns=['Doc_ID','Matching_Pred_Class','Pred_Prob'])

    # Establish a For loop to loop over all documents
    for doc_id in range(len(train_corpus)):
        
        # Calculate Document Similarity Measures
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        sims = pd.DataFrame(sims, columns=['Pred_Class_Index','Prob'])
        
        # Merge Similarity Measures with Original Scores
        sims = pd.merge(sims, scores_join, on=['Pred_Class_Index','Pred_Class_Index'])
        sims = sims.sort_values(['Prob'], ascending=[False])
        
        #Select the second most similar document (otherwise it will match to self)
        pred_class = sims[target].iloc[1]
        pred_prob = sims['Prob'].iloc[1]
        
        # Drop any estimates/predictions that are below the probability threshold
        if pred_prob<limit:
            pred_class=0
        
        # Append the estimates to the estimates dataframe
        estimates = estimates.append({'Doc_ID':doc_id, 'Matching_Pred_Class':pred_class, 'Pred_Prob':pred_prob}, ignore_index=True)
            
    return(estimates)
    
    
def Doc2Vec_Estimates_Testing(model, train_corpus, scores_join, target, limit=0):
    '''
    Purpose: This function produces an estimated score for every document by attempting to identify the closest related/matching document to the target document.
             This requires the following steps:
                 
                 1) Initalize an empty estimates dataframe to hold estimated values in
                 2) For every document in the dataset find the most similar document within the corpus to the document provided (in comparison to the second most similar document as with training data)
                 3) Assign the value of the most similar document to this document as well as the "probabiliy"/"similarity" of the match
                 4) If the probabily falls below a given threshold (usually .7) assign the predicted class to 0
                 5) Append the result to the estimates dataframe
        
    Input: This function requires the following inputs:
        
            1) model = a gensim.models.doc2vec.Doc2Vec object (can be generated using the Doc2Vec_Training_Model function)
            2) train_corpus = a doc2vec.TaggedDocument list (returned by the read_corpus function)
            3) scores_join = a Pandas dataframe that contains the index of all documents within the training set as well as the target score for those documents
            4) target = the target/feature you are attmeping to model/estiamte
            5) limit = the probability limit that a prediction must exceed for that prediction to be assigned/passed on (usually .7, but defaults to 0 if not assigned)
    
    Output: This function returns a pandas DataFrame with the following features:
        
            1) Doc_Id = The index of provided document
            2) Matching_Pred_Class = The predicted class/value of that document (assigned using the method described above)
            3) Pred_Prob = The Doc2Vec provided probability of a match
        
    '''
    
    # Initialize the Dictionary to Return
    estimates = pd.DataFrame(columns=['Doc_ID','Matching_Pred_Class','Pred_Prob','Weighted_Class'])

    # Establish a For loop to loop over all documents
    for doc_id in range(len(train_corpus)):
        
        # Calculate Document Similarity Measures
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        sims = pd.DataFrame(sims, columns=['Pred_Class_Index','Prob'])
        
        # Merge Similarity Measures with Original Scores
        sims = pd.merge(sims, scores_join, on=['Pred_Class_Index','Pred_Class_Index'])
        sims = sims.sort_values(['Prob'], ascending=[False])
        
        #Select the second most similar document (otherwise it will match to self)
        pred_class = sims[target].iloc[0]
        pred_prob = sims['Prob'].iloc[0]
        
        # Drop any estimates/predictions that are below the probability threshold
        if pred_prob<limit:
            pred_class=0
        
        # Multiply the probability of a document match by the score of that document to get the average weighted class
        sims['Weighted_Class'] = sims.apply(lambda row: (row[target]*row['Prob']), axis=1)
        weighted_class = sims['Weighted_Class'].mean()
        
        # Append the estimates to the estimates dataframe
        estimates = estimates.append({'Doc_ID':doc_id, 'Matching_Pred_Class':pred_class, 'Pred_Prob':pred_prob}, ignore_index=True)
            
    return(estimates)


def Document_Matching_Training(train, doc, target, limit):
    '''
    Purpose: This function combines all the required document matching functions to produce target estimates for every document in a provided training set.
             It accomplishes this through the following steps/functions:
                 
                 1) Develops a training corpus for the Doc2Vec model using the read_corpus function
                 2) Trains a Doc2Vec model using the Doc2Vec_Training_Model function
                 3) Produces and returns target estimates for every document in the training set using the Doc2Vec_Estimates_Training function
        
    Input: This function requires the following inputs:
            
            1) train = a pandas DataFrame that contains the documents to be trained on and scored
            2) doc = the column name (string) that contains the original document text within the DataFrame
            3) target = the column name (string) that contains the score we are attempting to estimate
            4) limit = the probability limit that a prediction must exceed for that prediction to be assigned/passed on (usually .7, but defaults to 0 if not assigned)
        
    Output: This function returns three items in a dictionary:
        
            1) train_estimates = a pandas DataFrame that contains estimates for every document in the training set
            2) scores_join = a pandas DataFrame with the index and score of the original training data (used as an input to other functions)
            3) gensim_model = the Doc2Vec gensim model developed using the tagged training data
    
    '''
    
    # Develop the training corpus
    train_corpus = list(read_corpus(train, doc))
    
    # Obtain Estimates
    gensim_training = Doc2Vec_Training_Model(train_corpus, doc)
    
    scores_join = pd.DataFrame(train[target])
    scores_join['Pred_Class_Index'] = scores_join.index
    train_esimates = Doc2Vec_Estimates_Training(gensim_training, train_corpus, scores_join, target, limit=0)
    
    return({'train_estimates':train_esimates, 'scores_join':scores_join, 'gensim_model':gensim_training})


def Document_Matching_Testing(test, doc, target, scores_join, gensim_training, limit):
    '''
    Purpose: This function combines all the required document matching functions to produce target estimates for every document in a provided testing set.
             It accomplishes this through the following steps/functions:
                 
                 1) Develops a testing corpus for the Doc2Vec model using the read_corpus function
                 2) Produces and returns target estimates for every document in the training set using the Doc2Vec_Estimates_Training model provided to the function
        
    Input: This function requires the following inputs:
            
            1) test = a pandas DataFrame that contains the documents to be scored
            2) doc = the column name (string) that contains the original document text within the DataFrame
            3) target = the column name (string) that contains the score we are attempting to estimate
            4) gensim_training = the gensim.models.doc2vec.Doc2Vec model (provided by the Document_Matching_Training function)
            4) limit = the probability limit that a prediction must exceed for that prediction to be assigned/passed on (usually .7, but defaults to 0 if not assigned)
        
    Output: This function returns a pandas DataFrame that contains estimates for every document in the training set
    '''

    # Develop the training corpus
    test_corpus = list(read_corpus(test, doc, target))
    
    # Obtain Estimates
    test_esimates = Doc2Vec_Estimates_Training(gensim_training, test_corpus, scores_join, target, limit=0)
    
    return(test_esimates)

