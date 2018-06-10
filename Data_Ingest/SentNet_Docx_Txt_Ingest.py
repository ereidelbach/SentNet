#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:29:34 2018

@author: Eric Reidelbach

:DESCRIPTION:
    - This script will ingest individual .docx files and .txt files and store
        them in a dataframe that matches the original format of the training
        data (i.e. training_set_rel3.csv in the \Data\ folder).
    - DataFrame variables dependo on the type of data being ingested but will
        include, at a minimum:
            * essay_id: A unique identifier for each individual student essay
            * essay_set: the set which the essay belongs to (1-8)
            * essay: The ascii text of a student's response
            * rater1_domain1: Rater 1's domain 1 score
            * rater2_domain1: Rater 2's domain 1 score
            * domain1_score: Resolved score between the raters 
                                (sum of all domain1 scores)
            * raterX_traitY: each rater may score specific traits about each essay

:REQUIRES:
    - This script requires the docx2txt package:
        https://github.com/ankushshah89/python-docx2txt
    - The package can be installed via the command:
        'pip install docx2txt
    
:TODO:
    NONE
"""
 
#==============================================================================
# Package Import
#==============================================================================
import docx2txt
import os   
import pandas as pd

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================
def ingest_files(path):
    '''
    Description:
        - This function will ingest every .docx file contained within the 
            specified folder (i.e. `path` )and store the contents in a Dataframe
    Input:
        - path (STRING):  the directory that contains the desired .docx data
        
    Output:
        - a DataFrame containing the information from all .docx files in the
            specified directory (i.e. `path`)
    '''
    file_list = []
    # go through every file in the specified path
    for filename in os.listdir(path):
        # create a dictionary for each document and a list for storing 
        #   pieces of each document during ingest
        doc_dict = {}
        doc_elements = []
        if filename.endswith('.docx'):
            # read in `docx` files
            doc = docx2txt.process(os.path.join(path, filename))
            # break up the contents of the file by splitting on line spaces
            doc_elements = doc.split('\n\n')
            # iterate throgh the elements two at a time to grab the variable name
            #   and the data associated with that variable (i.e. header)
            for (key, value) in zip(doc_elements[::2], doc_elements[1::2]):
                doc_dict[key] = value
        elif filename.endswith('.txt'):
            # read in `txt` files
            txtfile = open(os.path.join(path, filename), 'r',  encoding = 'utf-8')
            doc = txtfile.read()
            # break up the contents of the file by splitting on line returns
            doc_elements = doc.split('\n')
            for element in doc_elements:
                if element == '':
                    continue
                key = element.split(': ')[0]
                value = element.split(': ')[1]
                doc_dict[key] = value
        else:
            # account for different file types
            print ('Unknown file type detected.  SentNet can only ingest .txt'\
                   ', .docx, .xlsx, .xls, or .csv file types.')
            continue
        # store the dictionary in a list
        file_list.append(doc_dict)        
    # return a dataframe version of the list
    return pd.DataFrame(file_list)

#==============================================================================
# Working Code
#==============================================================================

# Read in the contents of the specified path to a Pandas DataFrame
df = ingest_files(r'E:\Projects\SentNet\Data\Set7\docx')