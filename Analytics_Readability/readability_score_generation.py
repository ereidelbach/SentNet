#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:42:29 2018

@author: Eric Reidelbach

:DESCRIPTION:
    - The purpose of this script is to generate Interpretability and 
        Readability scores for text documents based on the following three indices:
            1. Flesch-Kincaid
            2. Gunning Fog
            3. Cloeman-Liau

:REQUIRES:
            
:TODO:
"""
 
#==============================================================================
# Package Import
#==============================================================================
import os   
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('cmudict')
from textatistic import Textatistic
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer\
from nltk.corpus import cmudict



#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================
def count_num_syllables(document):
    syllable_count = 0
    for word in document.split():
        count = 0
        vowels = 'aeiouy'
        word = word.lower().strip(".:;?!")
        if word[0] in vowels:
            count +=1
        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count +=1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count+=1
        if count == 0:
            count +=1
        syllable_count += count
    return syllable_count

'''
    The Flesch Readability score is calculated as follows:
    206.835 - [1.015 * (total words/total sentences)] - [84.6 * (total syllabes/total words)]
'''
def FleschReadability(document):  
    # Determine the document's word count
    count_word = len(document.split())
#    tokenizer = RegexpTokenizer(r'\w+')
#    count_word2 = len(tokenizer.tokenize(document))
    
    # Determine the document's sentence count
    count_sentence = len(nltk.sent_tokenize(document))
    
    # Determine the document's syllable count
    count_syllables = count_num_syllables(document)

    
    # Calculate the Flesch Reading score
    return (206.835 - (1.015 * (count_word/count_sentence)) - (84.6*(count_syllables/count_word)))
    
    
def FleschKincaidGradeLevel(document):
    pass


#==============================================================================
# Working Code
#==============================================================================

# Set the project working directory
os.chdir(r'E:\Projects\SentNet\Data')

# Read in Training Data for scoring
df = pd.read_csv('training_set_rel3.csv', encoding = "ISO-8859-1")
document_list = list(df['essay'])

# Compute Flesch readability ease
FleschReadability(document_list[0])


s = Textatistic(document_list[0])
s.counts
s.flesch_score

# Compue Flesch-Kincaid grade level
# [0.39 * (total words / total sentences)] + [11.8 + (total syllabes / total words)] - 15.59
