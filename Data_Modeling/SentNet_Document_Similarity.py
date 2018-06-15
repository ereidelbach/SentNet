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
# Part #1 - Development/Testing 
# =============================================================================

'''
# Read in the train and test data
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))


# Split into Train and Test sets for now
train, test = train_test_split(data, test_size=0.2)

# Clean the text - DOES THIS WORK BETTER??? Need to test
#train['essay_cleaned'] =  train.apply(lambda row: clean_sentence(row['essay']), axis=1)
#test['essay_cleaned'] = test.apply(lambda row: clean_sentence(row['essay']), axis=1)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# Read and Preprocess Text
def read_corpus(data, field, tokens_only=False):
    for index, row in data.iterrows():
        if tokens_only:
            yield gensim.utils.simple_preprocess(row[field])
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]), [index])
                

#train_corpus = list(read_corpus(train, 'essay_cleaned'))
#test_corpus = list(read_corpus(test, 'essay_cleaned', tokens_only=True))

train_corpus = list(read_corpus(train, 'essay'))
test_corpus = list(read_corpus(test, 'essay', tokens_only=True))

# Training the gensim model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=4, epochs=100)
model.build_vocab(train_corpus)
%time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# Assessing Model
ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])

collect = collections.Counter(ranks)

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

'''
# =============================================================================
# Part #1 - Document Matching  
# =============================================================================

# Select the target that you want to estimate
target = 'rater1_domain1'

# Defining Functions for Document Matching            
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
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[field]), [index])

# Defining the function to train and return the Gensim Model
def Doc2Vec_Training_Model(train_corpus, field, target):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Preprocess text for Doc2Vec Model
    
    # Train the Model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=1, epochs=10)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    return(model)
    
# Defining the function to return the estimated probabilities for each document
def Doc2Vec_Estimates_Training(model, train_corpus, scores_join, target, limit=0):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
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
        sims = sims.sort_values(['Prob'], ascending=[False])
        
        #Select the second most similar document (otherwise it will match to self)
        pred_class = sims[target].iloc[1]
        pred_prob = sims['Prob'].iloc[1]
        
        if pred_prob<limit:
            pred_class=0
        # Obtain the score of the most similar document, save as the predicted class
        #pred_class = sims[target].iloc[sims['Prob'].argmax()]
        #pred_prob = sims['Prob'].iloc[sims['Prob'].argmax()]
        
        # Multiply the probability of a document match by the score of that document to get the average weighted class
        # sims['Weighted_Class'] = sims.apply(lambda row: (row[target]*row['Prob']), axis=1)
        # weighted_class = sims['Weighted_Class'].mean()
        
        # Append the estimates to the estimates dataframe
        estimates = estimates.append({'Doc_ID':doc_id, 'Pred_Class':pred_class, 'Pred_Prob':pred_prob}, ignore_index=True)
            
    return(estimates)
    
def Doc2Vec_Estimates_Testing(model, train_corpus, scores_join, target, limit=0):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
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
        sims = sims.sort_values(['Prob'], ascending=[False])
        
        #Select the second most similar document (otherwise it will match to self)
        pred_class = sims[target].iloc[0]
        pred_prob = sims['Prob'].iloc[0]
        
        if pred_prob<limit:
            pred_class=0
        # Obtain the score of the most similar document, save as the predicted class
        #pred_class = sims[target].iloc[sims['Prob'].argmax()]
        #pred_prob = sims['Prob'].iloc[sims['Prob'].argmax()]
        
        # Multiply the probability of a document match by the score of that document to get the average weighted class
        sims['Weighted_Class'] = sims.apply(lambda row: (row[target]*row['Prob']), axis=1)
        weighted_class = sims['Weighted_Class'].mean()
        
        # Append the estimates to the estimates dataframe
        estimates = estimates.append({'Doc_ID':doc_id, 'Pred_Class':pred_class, 'Pred_Prob':pred_prob}, ignore_index=True)
            
    return(estimates)

def Document_Matching_Training(train, doc, target, limit):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Develop the training corpus
    train_corpus = list(read_corpus(train, doc, target))
    
    # Obtain Estimates
    gensim_training = Doc2Vec_Training_Model(train_corpus, doc, target)
    
    scores_join = pd.DataFrame(train[target])
    scores_join['Pred_Class_Index'] = scores_join.index
    train_esimates = Doc2Vec_Estimates_Training(gensim_training, train_corpus, scores_join, target, limit=0)
    
    return({'train_estimates':train_esimates, 'scores_join':scores_join, 'gensim_model':gensim_training})


def Document_Matching_Testing(test, doc, target, scores_join, gensim_training, limit):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''

    # Develop the training corpus
    train_corpus = list(read_corpus(train, doc, target))
    
    test_esimates = Doc2Vec_Estimates_Training(gensim_training, train_corpus, scores_join, target, limit=0)
    
    return(test_esimates)

        
'''
pred_crosstab = pd.crosstab(train_esimates["Pred_Class"],train[target])
print(pred_crosstab)

# Testing
test_corpus = list(read_corpus(test, 'essay', target))

%time test_estimates = Doc2Vec_Training_Features(gensim_training, test_corpus, scores_join, target)

pred_crosstab = pd.crosstab(test_esimates["Pred_Class"],test[target])
print(pred_crosstab)
'''

# =============================================================================
# Part #2 - Document Similarity  
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
'''