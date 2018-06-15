#!/usr/bin/env python3.6
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
    - Python-Docx package (https://python-docx.readthedocs.io/en/latest/)
        * install via the command:  'pip install python-docx'
        * call library via the command: 'import docx'
    
:TODO:
    NONE
"""

#==============================================================================
# Package Import
#==============================================================================
import os  
import pandas as pd 
import docx
from pathlib import Path

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================
def convert2docx(data, col_names, setnum):
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
    # create the .docx file
    doc = docx.Document()
    
    # output the contents of each column to the .docx file
    for i in range(0,len(data)):
        doc.add_heading(col_names[i], 2)
        doc.add_paragraph(str(data[i]))

    # check to make sure a folder exists in the target directory for the set
    dir_path = Path('Data',setnum)
    if os.path.isdir(dir_path) == False:
        os.makedirs(dir_path)
        
    # check to make sure a `docx` folder exists in the target directory
    dir_path = Path(dir_path, 'docx')
    if os.path.isdir(Path(dir_path)) == False:
        os.makedirs(Path(dir_path))
    
    # save the created .docx file
    doc.save(str(Path(dir_path, str(data['essay_id']) + '.docx')))

def convert2txt(data, col_names, setnum):
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
    # check to make sure a folder exists in the target directory for the set
    dir_path = Path('Data',setnum)
    if os.path.isdir(dir_path) == False:
        os.makedirs(dir_path)        

    # check to make sure a `txt` folder exists in the target directory    
    if os.path.isdir(Path(dir_path, 'txt')) == False:
        os.makedirs(Path(dir_path, 'txt'))
    
    filename = str(Path(dir_path, 'txt', str(data['essay_id']) + '.txt'))
    with open(filename, "w", encoding = 'utf-8') as my_output_file:
        for i in range(0, len(data)):
            my_output_file.write(col_names[i] + ": " + str(data[i]) + '\n')
    my_output_file.close()

#==============================================================================
# Working Code
#==============================================================================

# Set the project working directory
#os.chdir(r'E:\Projects\SentNet\')
#os.chdir(r'C:\MSA\Projects\SentNet\')
os.chdir('/home/ejreidelbach/projects/SentNet')

# Read in Training Data for scoring
df = pd.read_csv(Path('Data','training_set_rel3.csv'), encoding = 'windows-1252')

# For every essay set in the data, extract the data for that set
#   and create an individual file for every row in that set
essay_set_list = df['essay_set'].value_counts().sort_index().index.tolist()
for df_num in essay_set_list:
    # create the subset of the original data based on the `essay_id`
    df_sub = df[df['essay_set'] == df_num]
    df_sub.reset_index(inplace=True)
    del df_sub['index']
    df_sub = df_sub.dropna(thresh=0.8*len(df_sub), axis=1)
    
    # create file names and folder names for the new data
    file = 'Set' + str(df_num)
    filename = (str(Path('Data','Set' + str(df_num), str(
            'training_set_rel3_set' + str(df_num) + '.csv'))))

    # create the individual .docx and .txt files for every row in the subset
    columns = list(df_sub.columns)
    for index, row in df_sub.iterrows():
        data = row
        convert2docx(row, columns, file)
        convert2txt(row, columns, file)
        if index%100 == 0:
            print('Finished with: ' + str(index) + ' files from ' + file)
            
    # create a .csv of the subset dataframe w/ only the essays in that id group
    df_sub.to_csv(filename, encoding = 'utf-8-sig', index='False')

# We're going to focus on set 7 and set 8 as that has the most information (i.e. scores)
#   Extract sets 7 and 8 and remove any columns that don't have relevant info
#df7 = df[df['essay_set'] == 7]
#df7.reset_index(inplace=True)
#del df7['index']
#df7 = df7.dropna(thresh=0.8*len(df7), axis=1)
#df7.to_csv('Set7/training_set_rel3_set7.csv', encoding = 'utf-8-sig', index='False')
#
#df8 = df[df['essay_set'] == 8]
#df8.reset_index(inplace=True)
#del df8['index']
#df8 = df8.dropna(thresh=0.8*len(df8), axis=1)
#df8.to_csv('Set8/training_set_rel3_set8.csv', encoding = 'utf-8-sig', index='False')
#
## Write the contents of Set 7 to individual .docx and .txt documents
#columns = list(df7.columns)
#for index, row in df7.iterrows():
#    convert2docx(row, columns, 'Set7')
#    convert2txt(row, columns, 'Set7')
#    if index%100 == 0:
#        print('Finished with: ' + str(index) + 'files from Set7')
#
## Write the contents of Set 8 to individual .docx and .txt documents
#columns = list(df8.columns)    
#for index, row in df8.iterrows():
#    convert2docx(row, columns, 'Set8')
#    convert2txt(row, columns, 'Set8')
#    if index%100 == 0:
#        print('Finished with: ' + str(index) + 'files from Set8')