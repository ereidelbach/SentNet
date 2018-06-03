#!/usr/bin/env python3.6.4
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:17:49 2018

@author: Eric

:DESCRIPTION:
    - This script will attempt to make a basic UI capable of selecting files
        within a folder and sending them to readability score functions for 
        processing (the readability score functions are contained in the 
        `Analytics_Readability` folder in the file:
            `readability_score_generation_packages.py`)
    
:REQUIRES:
    
:TODO:
"""

#==============================================================================
# Package Import
#==============================================================================
import os   
import tkinter

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================
'''
    Description:
        
    Input:
        
    Output:
'''    

#==============================================================================
# Working Code
#==============================================================================

# Set the project working directory
os.chdir(r'C:\MSA\Projects\SentNet\Data')

root = tkinter.Tk()
root.title('SentNet: Readability Score System')

button1 = tkinter.Button(root, text='button1')
button2 = tkinter.Button(root, text='button2')

text = tkinter.Entry(root)
listbox = tkinter.Listbox(root)

text.pack()
button1.pack()
button2.pack()
listbox.pack()

root.mainloop()