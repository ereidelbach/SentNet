#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:DESCRIPTION:
    - This script brings together the functionality of SentNet into one
        easy-to-run file that showcases the capabilities of the prototype in 
        a very structured environment. 
    - This file will unpack the entire contents of the `training_set_rel3.csv` 
        essay set into individual .docx files in the Data folder.
    - It will then inject pre-selected images into each file at random to 
        simulate visual graphics in IC analytic products
    - Finally, it will construct a model for the contents of the 
        `Data/Set1/docx` folder
        ** this set can easily be changed to any of the eight sets by modifying
            the variable `data_set` in the code below

:REQUIRES:
    NONE
   
:TODO:
    NONE
"""
 
#==============================================================================
# Package Import
#==============================================================================
from Data_Ingest.SentNet_Convert_File_Types import unpack_training_files
from Data_Ingest.SentNet_Insert_Images_Into_Data import inject_training_files
from Data_Modeling.SentNet_Training_Master_Functionalized import model_training_files

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================

#==============================================================================
# Working Code
#==============================================================================
'''
Unpack the contents of the file `training_set_rel3.csv,` located in the Data 
folder, into individual .docx files.
'''
unpack_training_files()

'''
Inject the newly unpacked .docx files with random images from the Data/Images 
folder to simulate the use of visual graphics in typical IC analytic products.
'''
inject_training_files()

'''
8 data sets are available for use in the training data (1 through 8) for the 
initial walkthrough. We'll use Set1 but feel free to change the variable below 
to specify any available set for testing (e.g., Set2, Set3, Set4, etc.).
'''
data_set = 'Set1'

'''
Create and test a scorecard model based for the data_set specified above
    - The resulting model results will be saved to the Model_Results folder 
        as an .xlsx file
    - The resulting model will be exported to the Model_Files folder 
        as a .pkl file
    - The resulting feature set will be saved to the Model_Files folder 
        as a .csv file

NOTE: The Document Matching and Document Similarity functions can take several 
minutes to run.  In total, the model building process will average roughly 
30 minutes for a typical set folder.
'''
#model_training_files(data_set)