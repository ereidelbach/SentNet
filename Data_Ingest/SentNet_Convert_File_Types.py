#!/usr/bin/env python3.6.4
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:53:55 2018

@author: Eric

:DESCRIPTION:
    - The purpose of this script is to break up the Kaggle training set and
        validation set data into individual .docx and .txt documents to
        prove the validity of our proposed ingest capabilities.
    - This script will create duplicates copies (1 .txt and 1 .docx) for every
        row in the .xlsx files.
    
:REQUIRES:
    - Python-Docx library (install python-docx)
    
:TODO:
"""

#==============================================================================
# Package Import
#==============================================================================
import os  
import pandas as pd 
import docx

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================
'''
    Description:
        - This function will take the passed in string from a row in a .xlsx
            or .csv file and create a new .docx file containing only this string        
    Input:
        - data (STRING):  a string containing the full contents of an essay
            submission along with any additional scoring information for that
            essay
        
    Output:
        - a .docx file will be written to the /Data/txt subfolder within the
            main project working folder
'''    
def convert2docx(data, col_names, setnum):
    # create the .docx file
    doc = docx.Document()
    
    # output the contents of each column to the .docx file
    for i in range(0,len(data)):
        doc.add_heading(col_names[i], 2)
        doc.add_paragraph(str(data[i]))
    
    # save the created .docx file
    doc.save(setnum + '/docx/' + str(data['essay_id']) + '.docx')

'''
    Description:
        - This function will take the passed in string from a row in a .xlsx
            or .csv file and create a new .txt file containing only this string
        
    Input:
        - data (STRING):  a string containing the full contents of an essay
            submission along with any additional scoring information for that
            essay
        
    Output: 
        - a .txt file will be written to the /Data/txt subfolder within the
            main project working folder
'''    
def convert2txt(data, col_names, setnum):
    filename = setnum + '/txt/' + str(data['essay_id']) + '.txt'
    with open(filename, "w", encoding = 'utf-8') as my_output_file:
        for i in range(0, len(data)):
            my_output_file.write(col_names[i] + ": " + str(data[i]) + '\n')
    my_output_file.close()

#==============================================================================
# Working Code
#==============================================================================

# Set the project working directory
# Set the project working directory
#os.chdir(r'E:\Projects\SentNet\Data')
os.chdir(r'C:\MSA\Projects\SentNet\Data')

# Read in Training Data for scoring
df = pd.read_csv('training_set_rel3.csv', encoding = "ISO-8859-1")

# We're going to focus on set 7 and set 8 as that has the most information (i.e. scores)
#   Extract sets 7 and 8 and remove any columns that don't have relevant info
df7 = df[df['essay_set'] == 7]
df7 = df7.dropna(thresh=0.8*len(df7), axis=1)
df7.to_csv('Set7/training_set_rel3_set7.csv', index='False')

df8 = df[df['essay_set'] == 8]
df8 = df8.dropna(thresh=0.8*len(df8), axis=1)
df8.to_csv('Set8/training_set_rel3_set8.csv', index='False')

# Write the contents of Set 7 to individual .docx and .txt documents
columns = list(df7.columns)
for index, row in df7.iterrows():
    convert2docx(row, columns, 'Set7')
    convert2txt(row, columns, 'Set7')
    if index%100 == 0:
        print(index)

# Write the contents of Set 8 to individual .docx and .txt documents
columns = list(df8.columns)    
for index, row in df8.iterrows():
    convert2docx(row, columns, 'Set8')
    convert2txt(row, columns, 'Set8')
    if index%100 == 0:
        print(index)