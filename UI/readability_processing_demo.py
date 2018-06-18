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
    - following this tutorial: 
        https://likegeeks.com/python-gui-examples-tkinter-tutorial/
    - Documentation:
        https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Combobox
        (OLD) http://effbot.org/tkinterbook/tkinter-classes.htm
:TODO:
"""

#==============================================================================
# Package Import
#==============================================================================
from pathlib import Path
import os   
import tkinter as tk
from tkinter.ttk import *

#==============================================================================
# Function Definitions / Reference Variable Declaration
#==============================================================================
def clicked():
    '''
    Description:
        
    Input:
        
    Output:
    ''' 
    lbl2.configure(text="Files selected")

def clicked_analyze():
    '''
    Description:
        
    Input:
        
    Output:
    ''' 
    lbl2.configure(text="Analyze the selected documents")

def clicked_scorecard():
    '''
    Description:
        
    Input:
        
    Output:
    ''' 
    lbl2.configure(text="Create a New RSEATS Scorecard")


def action_detect_files():
    '''
    Description:
        - this is a precursor to fulfilling RSEATS import requirement 1:
            1.) Import
        - this function checks to see if files have already been created
            for analysis
    Input:
        
    Output:
    '''    
    pass
    
def action_create_files():
    '''
    Description:
        - this fulfills RSEATS import requirements 1 & 3:
            1.) Import
            3.) Document Cleaning
    Input:
        
    Output:
    '''
    pass

def action_analyze_files():
    '''
    Description:
        - this fulfills RSEATS import requirements 4, 5, 6 and 7
            4.) WordNet Normalization
            5.) Construct Graph
            6.) Extract Synset Graph Features
            7.) Document Similarity Scoring
        
    Input:
        
    Output:
    '''
    pass

#==============================================================================
# Working Code
#==============================================================================

# Set the project working directory
os.chdir(Path("E:\Projects\SentNet"))
#os.chdir(Path('/home','ejreidelbach','projects','SentNet'))

# scan for existing file folders
set_folders = [x for x in os.listdir(Path('Data')) if 'Set' in x]
set_folders = sorted(set_folders)
# scan for files in the existing folder
doc_folders = []
for set in set_folders:
    # retrieve all file names and remove the '.docx' extension for all documents
    #   in the folder but ignore any possible sub-directories
    doc_list = [x.split('.docx')[0] for x in os.listdir(
            Path('Data', set, 'docx', 'Images')) if os.path.isfile(
                    Path('Data', set, 'docx', 'Images',x))]
    # convert the file names to int form to enable proper sorting
    doc_list = [int(x) for x in doc_list]
    # sort the document list
    doc_list = sorted(doc_list)
    # add the '.docx' extenion back to every filename
    doc_list = [str(x) + '.docx' for x in doc_list]
    # add the filename to the final document list
    doc_folders.append(doc_list)


############################ GUI CONFIG
# create the program
window = tk.Tk()
window.geometry('800x400')
window.title("Welcome to SentNet: An RSEATS Analysis Tool")

############################ DOCUMENT SELECTION
# create a label
lbl = tk.Label(window, text="Analyze Document",
               font=("Verdana", 16), wraplength=600)
# must specify a grid for the label to show up
lbl.grid(column = 0, row = 0)

# create the choose file button
btn = tk.Button(window, text="Choose File", bg="gray", fg="black", command=clicked)
btn.grid(column=0, row=1)

# create a label to test our `choose file` button
lbl2 = tk.Label(window, text="No file selected",
               font=("Verdana", 16), wraplength=600, relief="groove")
lbl2.grid(column = 1, row = 1)

# create a scorecard dropdown widget (combobox)
combo = tk.ttk.Combobox(window, values=set_folders)
combo.current(0)
combo.grid(row=2)

# create a large submit button
btn = tk.Button(window, text = "Submit", bg="blue", fg="white", command=clicked_analyze)
btn.grid(row=3)

############################ SCORECARD SELECTION
# create a label
lbl = tk.Label(window, text="Create New Analytic Score Card (Admin Users)",
               font=("Verdana", 16), wraplength=600)
# must specify a grid for the label to show up
lbl.grid(column = 0, row = 4)

# create a button that will create `Create a New RSEATS scorecard`
btn = tk.Button(window, text = "Submit", bg="blue", fg="white", command=clicked_scorecard)
btn.grid(row=5)

# Run the program
window.mainloop()

# On exit, destroy the program
window.destroy()