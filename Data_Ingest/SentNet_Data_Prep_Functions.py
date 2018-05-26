# -*- coding: utf-8 -*-
"""
Created on Mon May 21 23:02:05 2018

@author: GTayl
"""

# Import the required packages
from os import listdir
from os.path import isfile, join
import zipfile
import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
import os
import sys
import pandas as pd

def File_Selector_Training(path):
    '''
    Input: Path to a given folder/repository
    Output: Returns a list of files in the provided directory that end in .tsv, .csv, or .xlsx.
    
    Purpose: This allows for the scored training data to be read in from one or multuiple files.
             Data in these sheets must be provided in the accepted format (example spreadsheet has been provided).
             
             Col_1_UUID, Col_2_ScoreCard_Name, Col_3_Doc_Text, Col_4_List_of_JPEGs, Col_5_Criteria_1 .... Col_N_Criteria_N
    '''
    
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.tsv') or f.endswith('.csv') or f.endswith('.xlsx')]
    
    return(onlyfiles)



def Text_File_Selector(path):
    '''
    Input: Path to a given folder/repository
    Output: Returns a list of files in the provided directory that end in .txt, .doc, or .docx.
    
    Purpose: This allows testing data to be read in from one or multuiple files.
             
    '''
    
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.txt') or f.endswith('.doc') or f.endswith('.docx')]
    
    return(onlyfiles)



def Image_File_Selector(path):
    '''
    Input: Path to a given folder/repository
    Output: Returns a list of files in the provided directory that end in .txt, .doc, or .docx.
    
    Purpose: This allows testing data to be read in from one or multuiple files.
             
    '''
    
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    
    return(onlyfiles)



def Docx_File_Selector(path):
    '''
    Input: Path to a given folder/repository
    Output: Returns a list of files in the provided directory that end in .txt, .doc, or .docx.
    
    Purpose: This allows testing data to be read in from one or multuiple files.
             
    '''
    
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.docx')]
    
    return(onlyfiles)


def Extract_Docx_Images_Only(doc_folder_path, file_name, image_folder_path):
    '''
    Input: This function requires the following inputs:
        
            1. doc_folder_path = path to the folder that contains the given docx
            2. file_name = the file name of the docx to extract images from
            3. image_folder_path = the target folder images are to be deposited in
            
    Output: This function returns the following output:
        
            1. Images from the selected document are placed in the target folder (JPEGs only) with coresponding title and image number
            2. A list is returned of the image titles that have been extracted and stored from the given document
        
    Purpose: To assist with the preprocessing of documents by storing pictures for future analysis (perceptual hashing)
    '''
    # Concantenate strings to form full file path
    file_path = str(doc_folder_path)+"\\"+str(file_name)
    
    # Zip the selected file to break it into it's component parts
    z = zipfile.ZipFile(file_path)
    all_files = z.namelist()
    
    # Get all files in word/media/ directory
    images = list(filter(lambda x: x.startswith('word/media/'), all_files))
    
    # If there aren't any images in the document return an empty list
    if len(images)==0:
        return([])
    
    # Else save the images to the specified folder and return a list of images saved
    else:
        image_list = []
        n = 0
        for i in images:
            n += 1
            image_name = str(file_name)[:-5]+"_image_"+str(n)+".jpeg"
            image_file_path = image_folder_path+"\\"+image_name
            save_image = z.open(i).read()
            f = open(image_file_path,'wb')
            f.write(save_image)
            f.close()
            image_list.append(image_name)
            
    # Return the list of images that have been extracted and stored from this document
    return(image_list)
    

def xml2text(xml):
    """
    Input: XML object
    Output: String representation of that XML
    
    Purpose: A string representing the textual content of this run, with content
    child elements like ``<w:tab/>`` translated to their Python
    equivalent.
    
    Source:  https://github.com/ankushshah89/python-docx2txt/blob/master/docx2txt/
    """
    text = u''
    root = ET.fromstring(xml)
    for child in root.iter():
        if child.tag == qn('w:t'):
            t_text = child.text
            text += t_text if t_text is not None else ''
        elif child.tag == qn('w:tab'):
            text += '\t'
        elif child.tag in (qn('w:br'), qn('w:cr')):
            text += '\n'
        elif child.tag == qn("w:p"):
            text += '\n\n'
    return text


def qn(tag):
    """
    Input: 
    Output: 
    
    Purpose: Stands for 'qualified name', a utility function to turn a namespace
    prefixed tag name into a Clark-notation qualified tag name for lxml. For
    example, ``qn('p:cSld')`` returns ``'{http://schemas.../main}cSld'``.
    
    Source:  https://github.com/ankushshah89/python-docx2txt/blob/master/docx2txt/
    """
    prefix, tagroot = tag.split(':')
    uri = nsmap[prefix]
    return '{{{}}}{}'.format(uri, tagroot)

    
def Extract_Docx_Features(doc_folder_path, file_name, img_dir):
    """
    Input: 
    Output: 
    
    Purpose: 
    
    Adapted from: https://github.com/ankushshah89/python-docx2txt/blob/master/docx2txt/
    """
    file_path = str(doc_folder_path)+"\\"+str(file_name)
    
    text = u''
    image_list = []

    # unzip the docx in memory
    zipf = zipfile.ZipFile(file_path)
    filelist = zipf.namelist()

    # get header text
    # there can be 3 header files in the zip
    header_xmls = 'word/header[0-9]*.xml'
    for fname in filelist:
        if re.match(header_xmls, fname):
            text += xml2text(zipf.read(fname))

    # get main text
    doc_xml = 'word/document.xml'
    text += xml2text(zipf.read(doc_xml))

    # get footer text
    # there can be 3 footer files in the zip
    footer_xmls = 'word/footer[0-9]*.xml'
    for fname in filelist:
        if re.match(footer_xmls, fname):
            text += xml2text(zipf.read(fname))

    if img_dir is not None:
        # extract images
        for fname in filelist:
            _, extension = os.path.splitext(fname)
            if extension in [".jpg", ".jpeg", ".png", ".bmp"]:
                dst_fname = os.path.join(img_dir, str(file_name)[:-5]+"_"+os.path.basename(fname))
                with open(dst_fname, "wb") as dst_f:
                    dst_f.write(zipf.read(fname))
                    image_list.append(str(file_name)[:-5]+"_"+os.path.basename(fname))

    zipf.close()
    return({'file_name':file_name, 'text':text.strip(), 'image_list':image_list})


def Ingest_Training_Data(doc_folder_path, img_dir):
    '''
    Input: 
    Output: 
    
    Purpose: 
    '''
    
    # Initalize a dataframe to hold results
    Training_Data = pd.DataFrame(columns=['Doc_Title','Doc_Text','Doc_Images'])
    
    # Retrive all .docx files from the provided folder
    Docx_list = Docx_File_Selector(doc_folder_path)
    
    # Obtain data from documents and append to the Training Data Dataframe
    for d in Docx_list:
        temp_df = Extract_Docx_Features(doc_folder_path, d, img_dir)
        Training_Data = Training_Data.append({'Doc_Title':temp_df['file_name'], 'Doc_Text':temp_df['text'],'Doc_Images':temp_df['image_list']}, ignore_index=True)
    
    # Return the a data frame with the final features
    return(Training_Data)
    


        