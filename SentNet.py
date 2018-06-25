#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:54:11 2018

@author: ejreidelbach

:DESCRIPTION:

:REQUIRES:
   
:TODO:
"""
 
#==============================================================================
# Package Import
#==============================================================================
from Data_Ingest.SentNet_Convert_File_Types import unpack_training_files
from Data_Ingest.SentNet_Insert_Images_Into_Data import inject_training_files

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================

#==============================================================================
# Working Code
#==============================================================================

# Unpack the contents of the file `training_set_rel3.csv,` located in the
#   Data folder, into individual .docx files
unpack_training_files()

# Inject the newly unpacked .docx files with random images from the 
#   Data/Images folder to simulate the use of visual graphics in typical
#   IC analytic products
inject_training_files()