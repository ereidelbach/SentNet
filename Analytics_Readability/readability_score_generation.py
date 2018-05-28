#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:42:29 2018

@author: Eric Reidelbach

description

:REQUIRES:
    
:TODO:
    - The purpose of this script is to generate Interpretability and 
        Readability scores for text documents based on the following three indices:
            1. Flesch-Kincaid
            2. Gunning Fog
            3. Cloeman-Liau
"""
 
#==============================================================================
# Package Import
#==============================================================================
import os   
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================
def FleschReadability(document):
    # remove any words that are not from the English language
    doc_cleaned = 
    count_word = ""
    count_sentance = ""
    


def FleschKincaidGradeLevel(document):
    pass


#==============================================================================
# Working Code
#==============================================================================

# Set the project working directory
os.chdir(r'E:\Projects\SentNet\Data')

# Read in Training Data for scoring
df = pd.read_csv('training_set_rel3.csv', encoding = "ISO-8859-1")

# Compute Flesch readability ease

# Compue Flesch-Kincaid grade level
