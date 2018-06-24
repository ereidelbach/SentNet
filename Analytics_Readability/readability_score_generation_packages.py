#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
:DESCRIPTION:
    - The purpose of this script is to generate Interpretability and 
        Readability scores for text documents based on the following five indices:
            1. Dale-Chall
            2. Flesch
            3. Flesch-Kincaid
            2. Gunning Fog
            3. Smog  
    -The purpose of readability tests are to indicate how difficult a passage 
        of text is to understand or comprehend. Different tests have different 
        formulas but the general idea across all tests is the same: 
            provide a score based on characteristics such as average word 
            length or sentence length in order to assign a reading grade 
            level or a measurement of linguistic difficulty.  
            
:REQUIRES:
    - Textatistic Python Package
        https://pypi.org/project/textatistic/
    - TextStat Python Package
        https://pypi.org/project/textstat/       
:TODO:
"""
 
#==============================================================================
# Package Import
#==============================================================================
from textstat.textstat import textstat
from textatistic import Textatistic
#import spacy
#spacy.load('en')
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
        document_dict['tt_count_sent'] = s.sent_count
        document_dict['tt_count_sybl'] = s.sybl_count
        document_dict['tt_count_word'] = s.word_count
        document_dict['tt_score_dalechall'] = s.dalechall_score
        document_dict['tt_score_flesch'] = s.flesch_score
        document_dict['tt_score_fleschkincaid'] = s.fleschkincaid_score
        document_dict['tt_score_gunningfog'] = s.gunningfog_score
        document_dict['tt_score_smog'] = s.smog_score
    except(ZeroDivisionError):
        print('TEXTATISTIC -- Error with: ' + document)
        document_dict['tt_count_sent'] = 'N/A'
        document_dict['tt_count_sybl'] = 'N/A'
        document_dict['tt_count_word'] = 'N/A'
        document_dict['tt_score_dalechall'] = 'N/A'
        document_dict['tt_score_flesch'] = 'N/A'
        document_dict['tt_score_fleschkincaid'] = 'N/A'
        document_dict['tt_score_gunningfog'] = 'N/A'
        document_dict['tt_score_smog'] = 'N/A'      
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
    document_dict['ts_count_sent'] = textstat.sentence_count(document)
    document_dict['ts_count_sybl'] = textstat.syllable_count(document)
    document_dict['ts_count_word'] = textstat.lexicon_count(document)
    try:
        document_dict['ts_score_dalechall'] = textstat.dale_chall_readability_score(document)
        document_dict['ts_score_flesch'] = textstat.flesch_reading_ease(document)
        document_dict['ts_score_fleschkincaid'] = textstat.flesch_kincaid_grade(document)
        document_dict['ts_score_gunningfog'] = textstat.gunning_fog(document)
        document_dict['ts_score_smog'] = textstat.smog_index(document)
    except(ZeroDivisionError):
        print('TEXTSTAT -- Error with: ' + document)
        document_dict['ts_score_dalechall'] = 'N/A'
        document_dict['ts_score_flesch'] = 'N/A'
        document_dict['ts_score_fleschkincaid'] = 'N/A'
        document_dict['ts_score_gunningfog'] = 'N/A'
        document_dict['ts_score_smog'] = 'N/A'    
    return document_dict
        
#def textacy_scores(document):
#    '''
#    Description:
#        Function that calculates scores for all relevant readability metrics
#        utilizing the TEXTACY Python package.  
#           Note: Textacy requires the Spacy package to be installed as well
#    
#        Details on the package can be found here:
#            - https://chartbeat-labs.github.io/textacy/index.html
#            
#        You must install the English module for this package to function. 
#        It can be installed via the command:
#            "python -m spacy download en"
#       
#        Textacy can be installed via the Anaconda prompt on Windows:
#           "conda install -c conda-forge textacy"
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
#    document_dict['ty_count_sent'] = ts.n_sents
#    document_dict['ty_count_sybl'] = ts.n_syllables
#    document_dict['ty_count_word'] = ts.n_words
#    try:
#        document_dict['ty_score_colemanliau'] = ts.coleman_liau_index
#        document_dict['ty_score_dalechall'] = 'N/A'
#        document_dict['ty_score_flesch'] = ts.flesch_reading_ease
#        document_dict['ty_score_fleschkincaid'] = ts.flesch_kincaid_grade_level
#        document_dict['ty_score_gunningfog'] = ts.gunning_fog_index
#        document_dict['ty_score_smog'] = ts.smog_index
#    except(ZeroDivisionError):
#        print('TEXTACY -- Error with: ' + document)
#        document_dict['ty_score_colemanliau'] = 'N/A'
#        document_dict['ty_score_dalechall'] = 'N/A'
#        document_dict['ty_score_flesch'] = 'N/A'
#        document_dict['ty_score_fleschkincaid'] = 'N/A'
#        document_dict['ty_score_gunningfog'] = 'N/A'
#        document_dict['ty_score_smog'] = 'N/A'    
#    return document_dict
