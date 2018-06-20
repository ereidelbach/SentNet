#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:54:21 2018

@author: Eric Reidelbach

:DESCRIPTION:
    - The purpose of this script is to generate Interpretability and 
        Readability scores for text documents based on the following five indices:
            1. Dale-Chall
            2. Flesch
            3. Flesch-Kincaid
            2. Gunning Fog
            3. Smog  
            
:REQUIRES:
    - Textatistic Python Package
        https://pypi.org/project/textatistic/
    - TextStat Python Package
        https://pypi.org/project/textstat/
    - Textacy Python Package
        Requires the `en` library which can be installed via the command:
            `python -m spacy download en`
            ** Must run as administrator if on Windows
        Textacy can be installed via the Anaconda prompt on Windows:
            `conda install -c conda-forge textacy `
        
:TODO:
"""
 
#==============================================================================
# Package Import
#==============================================================================
import os
import pandas as pd
#import spacy
#spacy.load('en')
from textstat.textstat import textstat
from textatistic import Textatistic
#import textacy

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================
def textatistic_scores(document):
    '''
    Description:
        Function that calculates scores for all relevant readability metrics
        utilizing the TEXTATISTIC Python package.  
    
        Details on the package can be found here:
            - https://pypi.org/project/textatistic/
        
    Input:
        document (string): string containing the document to be scored
        
    Output:
        document_dict (dictionary): dictionary containing the resulting scores
            for the document along with counts for total number of words, 
            syllables, and sentences
    '''
    document_dict = {}
    try:
        s = Textatistic(document)
        document_dict['count_sent'] = s.sent_count
        document_dict['count_sybl'] = s.sybl_count
        document_dict['count_word'] = s.word_count
        document_dict['score_dalechall'] = s.dalechall_score
        document_dict['score_flesch'] = s.flesch_score
        document_dict['score_fleschkincaid'] = s.fleschkincaid_score
        document_dict['score_gunningfog'] = s.gunningfog_score
        document_dict['score_smog'] = s.smog_score
    except(ZeroDivisionError):
        print('TEXTATISTIC -- Error with: ' + document)
        document_dict['count_sent'] = 'N/A'
        document_dict['count_sybl'] = 'N/A'
        document_dict['count_word'] = 'N/A'
        document_dict['score_dalechall'] = 'N/A'
        document_dict['score_flesch'] = 'N/A'
        document_dict['score_fleschkincaid'] = 'N/A'
        document_dict['score_gunningfog'] = 'N/A'
        document_dict['score_smog'] = 'N/A'      
    return document_dict

def textstat_scores(document):
    '''
    Description:
        Function that calculates scores for all relevant readability metrics
        utilizing the TEXTSTAT Python package.  
    
        Details on the package can be found here:
            - https://pypi.org/project/textstat/
        
    Input:
        document (string): string containing the document to be scored
        
    Output:
        document_dict (dictionary): dictionary containing the resulting scores
            for the document along with counts for total number of words, 
            syllables, and sentences
    '''      
    document_dict = {}
    document_dict['count_sent'] = textstat.sentence_count(document)
    document_dict['count_sybl'] = textstat.syllable_count(document)
    document_dict['count_word'] = textstat.lexicon_count(document)
    try:
        document_dict['score_dalechall'] = textstat.dale_chall_readability_score(document)
        document_dict['score_flesch'] = textstat.flesch_reading_ease(document)
        document_dict['score_fleschkincaid'] = textstat.flesch_kincaid_grade(document)
        document_dict['score_gunningfog'] = textstat.gunning_fog(document)
        document_dict['score_smog'] = textstat.smog_index(document)
    except(ZeroDivisionError):
        print('TEXTSTAT -- Error with: ' + document)
        document_dict['score_dalechall'] = 'N/A'
        document_dict['score_flesch'] = 'N/A'
        document_dict['score_fleschkincaid'] = 'N/A'
        document_dict['score_gunningfog'] = 'N/A'
        document_dict['score_smog'] = 'N/A'    
    return document_dict

#def textacy_scores(document):
#    '''
#    Description:
#        Function that calculates scores for all relevant readability metrics
#        utilizing the TEXTACY Python package.  
#    
#        Details on the package can be found here:
#            - https://chartbeat-labs.github.io/textacy/index.html
#            
#        You must install the English module for this package to function:
#            - python -m spacy download en
#        
#    Input:
#        document (string): string containing the document to be scored
#        
#    Output:
#        document_dict (dictionary): dictionary containing the resulting scores
#            for the document along with counts for total number of words, 
#            syllables, and sentences
#    '''
#    document = document.encode('ascii', 'namereplace').decode('utf-8')
#    doc = textacy.Doc(document)
#    ts = textacy.TextStats(doc)
#    document_dict = {}
#    document_dict['count_sent'] = ts.n_sents
#    document_dict['count_sybl'] = ts.n_syllables
#    document_dict['count_word'] = ts.n_words
#    try:
#        document_dict['score_colemanliau'] = ts.coleman_liau_index
#        document_dict['score_dalechall'] = 'N/A'
#        document_dict['score_flesch'] = ts.flesch_reading_ease
#        document_dict['score_fleschkincaid'] = ts.flesch_kincaid_grade_level
#        document_dict['score_gunningfog'] = ts.gunning_fog_index
#        document_dict['score_smog'] = ts.smog_index
#    except(ZeroDivisionError):
#        print('TEXTACY -- Error with: ' + document)
#        document_dict['score_colemanliau'] = 'N/A'
#        document_dict['score_dalechall'] = 'N/A'
#        document_dict['score_flesch'] = 'N/A'
#        document_dict['score_fleschkincaid'] = 'N/A'
#        document_dict['score_gunningfog'] = 'N/A'
#        document_dict['score_smog'] = 'N/A'    
#    return document_dict
    
#==============================================================================
# Working Code
#==============================================================================

## Set the project working directory
#os.chdir(r'E:\Projects\SentNet\Data')
##os.chdir(r'C:\MSA\Projects\SentNet\Data')
#
## Read in Training Data Set 7 for scoring
#df = pd.read_csv('Set7/training_set_rel3_set7.csv', encoding = "utf-8",
#                 index_col = 0)
#
#document_list = list(df['essay'])
#
## Create a list for storing the scores of every document in this training set
#score_list = []
#
## Compute scores with both Textatistic and Textstat
#for document in document_list:
#    document_dict = {}
#    document_dict['text'] = document
#    
#    # retrieve the scores from all popular readability packages:
#    #   textatistic, textstat, and textacy
#    document_dict['textatistic'] = textatistic_scores(document)
#    document_dict['textstat'] = textstat_scores(document)
#    document_dict['textacy'] = textacy_scores(document)
#    
#    # store the results
#    score_list.append(document_dict)
#    
#    # count our progress in the list
#    if document_list.index(document) % 100 == 0:
#        print(document_list.index(document))
    