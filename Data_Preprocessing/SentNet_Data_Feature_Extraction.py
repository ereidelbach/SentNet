# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:17:02 2018

@author: GTayl
"""

####################################################################################
################################## Set-up ##########################################
####################################################################################
import time
start = time.time()
# Import the required packages
import pandas as pd
import numpy as np
import itertools
import sklearn
import networkx as nx

# Import required image hashing packages
from SentNet_Data_Prep_Functions import Ingest_Training_Data
from Goldberg_Perceptual_Hashing import ImageSignature
gis = ImageSignature()

# Importing NLTK related packages and content
try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tokenize import PunktSentenceTokenizer
    stop_words = set(stopwords.words('english'))
    
except:
    print(""" Note the English NLTK corpus will need to be downloaded for these to function properly. To download the corpus you must run the following:
            
            1) import nltk
            2) nltk.download()
            
            Select the following items to download:
            a) Stopwords
            b) WordNet
            c) WordNet_IC
            d) Punkt
               
            """)

# Import text analytics functions from readability_score_generation_packages.py
from readability_score_generation_packages import textatistic_scores, textstat_scores

        
# Define the list of punctuation to be removed
punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''

###############################################################################################
#################################### Data Ingest ##############################################
###############################################################################################

################################## .Docx Data Ingest ##########################################
'''
# Specify a folder that contains the training document in .docx format
doc_folder_path = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\Example_Docs"
img_dir = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Photos\\test_photos"

# Use the Ingest Training Data function from SentNet_Data_Prep_Functions.py to read in the training data 
data = Ingest_Training_Data(doc_folder_path, img_dir)

############################### Spreadsheet Data Ingest #######################################
# Alternatively read in data from a prepoulated spreadsheet (or database table)
traing_data = 'C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\training_set_rel3.tsv'
data = pd.DataFrame.from_csv(traing_data, sep='\t', header=0, encoding='ISO-8859-1')

# Select any subset (Scorecard) that you want to use for training
data = data[data['essay_set']==1]
print("Done Data Injest")
time1 = time.time()
print(time1-start)
'''
###############################################################################################
#################################### Data Cleaning ############################################
###############################################################################################
'''
In this section we define functions that will be used to clean/transform documents so they can be injested by the later feature generation functions.
'''

# Prep text
def remove_punc(text):
    '''
    Purpose: To remove punctuation marks/symbols from the provided string. Punctuation symbols to be removed are defined in the above set-up section.
        
    Input: String
        
    Output: String with puctuation symbols removed
    '''
    text_cleaned = ""
    for char in text:
       if char not in punctuations:
           text_cleaned = text_cleaned + char
    return(text_cleaned)
    
    
def clean_words(text):
    '''
    Purpose: To "clean" the provided sentence by removing punctuation and stop words. Punctuation symbols and stop words to be removed are defined in the above set-up section.
        
    Input: String
        
    Output: List of words (in order) with puctuation symbols and stop words removed.
    '''
    text = str(text).lower()
    text = remove_punc(text)
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return(filtered_sentence)

def clean_sentence(text):
    '''
    Purpose: To provide a cleaned/transformed string with punctuation marks and stop words removed.
        
    Input: String
        
    Output: String with puctuation marks and stop words removed
    '''
    string = ""
    temp = clean_words(text)
    for i in temp:
        if i ==".":
            string += i
        else:
            string += (" "+i)
    return(string)
 
###############################################################################################
########################## Document Features & Readability Tests ##############################
############################################################################################### 
'''
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''

# Create a copy of our original data pull to append to
def Readability_Features(data, target):
    
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    doc_readability_features = pd.DataFrame()
    
    # Extract measures for each document
    for index, row in data.iterrows():
        
        # Call feature generation functions
        temp_textatistic = pd.DataFrame(textatistic_scores(row[target]), index=[0])
        temp_textstat = pd.DataFrame(textstat_scores(row[target]), index=[0])
        #textacy = pd.DataFrame(textacy_scores(row['essay']), index=[0])
        
        # Join document features into a temporary dataframe
        temp_df_row = pd.concat([temp_textatistic,temp_textstat], axis=1, join='inner')
        
        # Append temporary dataframe to the master doc_readability_features dataframe
        doc_readability_features = doc_readability_features.append(temp_df_row, ignore_index=True)
    
    print("Done Readability Tests")
    return(doc_readability_features)

  
###############################################################################################
############################## Word Feature Extraction ########################################
############################################################################################### 

'''
Purpose: In this section we extract word counts from documents to use as the first feature class for our models.

Input: Pandas DataFrame with the document text stored in a column (in this case 'essay')
    
Output: A dense document (n) by word (w) (n x w) matrix that contains the frequency counts of "common words" in each document
        Selecting what constitutes a "common" word can be specified using the "limit" parameter below. 
        A limit parameter of 0.01 means that a word must appear in at least 1% of documents to be included in the term matrix.
        This helps to remove extreamly rare features (as well as mispellings) that will provide little value during the modeling process.
'''
def Word_Features(data, target, limit):
    
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Obtains a set of unique words in our corpus
    unqiue_words = set()
    
    for index, row in data.iterrows():
        temp_words = set(clean_words(row[target]))
        unqiue_words = unqiue_words.union(temp_words)
    
    # Sklearn Matrix Builder
        
    # Append a new column to our original dataframe with a "cleaned" version of the document text
    data['essay_clean'] = data.apply(lambda row: clean_sentence(row[target]),axis=1)
    
    # Define the vocab list for the Sklearn vecotrizer to search over
    unique_words = list(unqiue_words)
    
    # Train and run the sklearn vectorizor against the cleaned documents to get a sparse matrix representation
    count_vec = sklearn.feature_extraction.text.CountVectorizer(vocabulary=unique_words,lowercase=False)
    count_matrix = count_vec.fit_transform(data['essay_clean'])
    
    # Convert the sparse SciPy matrix into a dense matrix and convert intoa a pandas dataframe for further analysis
    count_matrix = pd.DataFrame(count_matrix.todense(),columns=unique_words)
    term_frequency_counts = pd.DataFrame(count_matrix.apply(lambda column: sum(column)),columns=['Counts'])
    
    # Subset the count_matrix to only inlcude terms that appear more than the specified limit
    selected_words = term_frequency_counts[term_frequency_counts['Counts']>limit]
    selected_words = selected_words.index
    word_matrix_features = count_matrix[selected_words]
    
    print("Done Word Features")
    return({'word_matrix_features':word_matrix_features, 'selected_words':selected_words})

###############################################################################################
############################# Synset Feature Generation #######################################
############################################################################################### 
'''
Purpose: In this section we extract synset counts from documents to use as the second feature class for our models.

Input: Pandas DataFrame with the document text stored in a column (in this case 'essay')
    
Output: A dense document (n) by synset (s) matrix (n x s) that contains the frequency counts of "common synsets" in each document
        Selecting what constitutes a "common" synset can be specified using the "limit" parameter below. 
        A limit parameter of 0.01 means that a synset must appear in at least 1% of documents to be included in the synset matrix.
        This helps to remove extreamly rare features that will provide little value during the modeling process.
'''
########################## Synset Translation Functions ####################################

# Return ths Hypernym code (as a string) for each hypernym in the path
def return_hypernym_codes(hypernym_list):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    hypernym_codes = []
    for hypernym in hypernym_list:
        hypernym_codes.append(str(hypernym.name()).replace(".","")) #removing periods for now
    return(hypernym_codes)


# Convert words into their hypernym code paths 
def WordNet_Translator(term):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''

    hypernym_list = []
    synset = wn.synsets(term)
    if len(synset)==0:
        #print(term) --> Enable this if you want to see the terms that do not have matching synsets
        pass
    else:
        synset = synset[0]
        hypernym_list = synset.hypernym_paths()
        while type(hypernym_list[0])==list:
            hypernym_list = hypernym_list[0]
        hypernym_list = return_hypernym_codes(hypernym_list)
    return(hypernym_list)


# Building a Synset Translation for each essay
punctuations = '''!()-[]{};.:'"\,<>/?@#$%^&*_~'''

def synset_translated_string(text):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    string = ""
    temp = clean_words(text)
    for i in temp:
        i = WordNet_Translator(i)
        for s in i:
            #s.replace(".","")
            string += (" "+s)
    return(string)


########################## Synset Translation & Extraction ####################################
def Synset_Features(data, target, limit): 
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Unique synsets in corpus
    unique_synsets = set() 
    
    for index, row in data.iterrows():
        temp_words = set(clean_words(row[target]))
        for word in temp_words:
            temp_synsets = WordNet_Translator(word)
            if len(temp_synsets)>0:
                unique_synsets = unique_synsets.union(temp_synsets)
            else:
                pass
    
    # Building a Synset Translation for each essay
    data['essay_synset_translation'] = data.apply(lambda row: synset_translated_string(row[target]),axis=1)
    
    # develop the vocab list for the methods to search over
    unique_synsets = list(unique_synsets)
    
    # Train and run the sklearn vectorizor to get a sparse matrix representation
    count_vec_syn = sklearn.feature_extraction.text.CountVectorizer(vocabulary=unique_synsets,lowercase=False,)
    count_matrix_syn = count_vec_syn.fit_transform(data['essay_synset_translation'])
    
    # Convert the sparse SciPy matrix into a dense matrix and convert intoa a pandas dataframe for further analysis
    count_matrix_syn = pd.DataFrame(count_matrix_syn.todense(),columns=unique_synsets)
    synset_frequency_counts = pd.DataFrame(count_matrix_syn.apply(lambda column: sum(column)),columns=['Counts'])
    
    # Define the minimum threshold to observe a item (word or synset) for it to be included in our analysis
    selected_synsets = synset_frequency_counts[synset_frequency_counts['Counts']>limit]
    selected_synsets = selected_synsets.index
    synset_matrix_features = count_matrix_syn[selected_synsets]
    
    print("Done Synset Generation")
    return({"synset_matrix_features":synset_matrix_features, "selected_synsets":selected_synsets})

###############################################################################################
############################ Word Graph Feature Generation ####################################
############################################################################################### 
'''
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''

######################### Word - Edge Feature Generation Functions ############################

sent_tokenize = nltk.tokenize.punkt.PunktSentenceTokenizer()

# Break document into sentences, then word tokens, return all combinations of tokens
def pd_edge_extractor(text):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    doc_edge_list = pd.DataFrame(columns=['Item_A','Item_B'])
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(clean_words(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list = doc_edge_list.append(pd.DataFrame(list(edge_list),columns=['Item_A','Item_B']), ignore_index=True)
    return(doc_edge_list)

def touple_edge_extractor(text):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    doc_edge_list = []
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(clean_words(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list.extend(edge_list)
        #print(len(doc_edge_list))
    return(doc_edge_list)

# Translate document text into edges representation
def edge_translation(text):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    doc_translation = ""
    doc_touples = touple_edge_extractor(text)
    for t in doc_touples:
        temp_t = str(t[0])+str(t[1])
        doc_translation = doc_translation +" "+temp_t
    return(doc_translation)

######################### Word - Edge Feature Extraction ############################
def Word_Edge_Features(data, target, limit):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Get the edge counts for all sentences in all documents
    master_edge_list = pd.DataFrame(columns=['Item_A','Item_B'])
    for index, row in data.iterrows():
        doc_edges = pd_edge_extractor(row[target])
        master_edge_list = master_edge_list.append(doc_edges, ignore_index=True)
    
    # Calculate which edges are the most common among the documents to collect counts for
    master_edge_list['edge_id'] = master_edge_list.Item_A.str.cat(master_edge_list.Item_B)
    selected_edge_list = pd.DataFrame(master_edge_list['edge_id'].value_counts())
    selected_edge_list = selected_edge_list[selected_edge_list['edge_id']>limit]
    selected_edge_list = list(selected_edge_list.index)
    
    # Translate document text into edges representation
    data['edge_translation']=data.apply(lambda row: edge_translation(row[target]), axis=1)
    
    # Train and run the sklearn vectorizor to get a sparse matrix representation
    count_vec_edges = sklearn.feature_extraction.text.CountVectorizer(vocabulary=selected_edge_list, lowercase=False)
    count_matrix_edges = count_vec_edges.fit_transform(data['edge_translation'])
    
    # Convert the sparse SciPy matrix into a dense matrix and convert intoa a pandas dataframe for further analysis
    edges_matrix_features = pd.DataFrame(count_matrix_edges.todense(), columns=selected_edge_list)
    
    print('Done with Word Edge Features')
    return({'edges_matrix_features':edges_matrix_features, 'master_edge_list':master_edge_list})


#################### Word - Betweeness Centrality Extraction #######################        

def Word_Centrality_Features(data, target, selected_words):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''

    # Create Betweeness Centrality Dataframe
    word_centrality_matrix = pd.DataFrame()
    
    # Populate DataFrame with betweenness centralities 
    for index, row in data.iterrows():
        
        # Extract all edges from the docuemnt as touples
        doc_edges = touple_edge_extractor(row['essay'])
        
        # Create a graph using all the edge inputs as touples
        G = nx.Graph()
        G.add_edges_from(doc_edges)
        btwn_dict = nx.betweenness_centrality(G)
        Node_list = set(G.nodes())
        Node_list = list(Node_list & set(selected_words))
        temp_dict = {}
        for n in Node_list:
            col = str(n)+"_btw"
            temp_dict[col]= btwn_dict[n]
        word_centrality_matrix = word_centrality_matrix.append(temp_dict,ignore_index=True)

    print("Done Word Graph Generation")
    return(word_centrality_matrix)

        
###############################################################################################
############################ Synset Graph Feature Generation ##################################
############################################################################################### 
'''
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''
   
######################### Synset - Edge Feature Generation Functions ###########################

#
def synset_translated_sentence(text):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    synset_list = []
    temp = clean_words(text)
    for i in temp:
        i = WordNet_Translator(i)
        for s in i:
            synset_list.append(s)
    return(synset_list)

#
def pd_edge_extractor_synset(text):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    doc_edge_list = pd.DataFrame(columns=['Synset_A','Synset_B'])
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(synset_translated_sentence(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list = doc_edge_list.append(pd.DataFrame(list(edge_list),columns=['Synset_A','Synset_B']), ignore_index=True)
    return(doc_edge_list)

#
def touple_edge_extractor_synset(text):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    doc_edge_list = []
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(synset_translated_sentence(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list.extend(edge_list)
        #print(len(doc_edge_list))
    return(doc_edge_list)

# Translate document text into a synset edges representation
def edge_translation_synset(text):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    doc_translation = ""
    doc_touples = touple_edge_extractor_synset(text)
    for t in doc_touples:
        temp_t = str(t[0])+str(t[1])
        doc_translation = doc_translation +" "+temp_t
    return(doc_translation)

######################### Synset - Edge Feature Extraction ############################
def Synset_Edge_Features(data, target, limit):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Get the synset edge counts for all sentences in all documents
    master_edge_list_synset = pd.DataFrame(columns=['Synset_A','Synset_B'])
    for index, row in data.iterrows():
        doc_edges_synset = pd_edge_extractor_synset(row[target])
        master_edge_list_synset = master_edge_list_synset.append(doc_edges_synset, ignore_index=True)
    
    # Calculate which synset edges are the most common among the documents to collect counts for
    master_edge_list_synset['edge_id'] = master_edge_list_synset.Synset_A.str.cat(master_edge_list_synset.Synset_B)
    selected_edge_list_synset = pd.DataFrame(master_edge_list_synset['edge_id'].value_counts())
    selected_edge_list_synset = selected_edge_list_synset[selected_edge_list_synset['edge_id']>limit]
    selected_edge_list_synset = list(selected_edge_list_synset.index)
    #master_edge_list_synset.drop(master_edge_list_synset.index, inplace=True) #Drop the dataframe here to preserve system resources
    
    # Translate document text into a synset edges representation
    data['edge_translation_synset']=data.apply(lambda row: edge_translation_synset(row[target]), axis=1)
    
    # Train and run the sklearn vectorizor to get a sparse matrix representation
    count_vec_edges_synset = sklearn.feature_extraction.text.CountVectorizer(vocabulary=selected_edge_list_synset, lowercase=False)
    count_matrix_edges_synset = count_vec_edges_synset.fit_transform(data['edge_translation'])
    
    # Convert the sparse SciPy matrix into a dense matrix and convert intoa a pandas dataframe for further analysis
    edges_matrix_features_synset = pd.DataFrame(count_matrix_edges_synset.todense(), columns=selected_edge_list_synset)

    print("Done with Synset_Edge_Features")
    return({'edges_matrix_features_synset':edges_matrix_features_synset, 'master_edge_list_synset':master_edge_list_synset})

#################### Synset - Betweeness Centrality Extraction #######################      

def Synset_Centrality_Features(data, target, selected_synsets):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Create Betweeness Centrality Dataframe
    synset_centrality_matrix = pd.DataFrame()
    
    # Populate DataFrame with betweenness centralities 
    for index, row in data.iterrows():
        
        # Extract all edges from the docuemnt as touples
        doc_edges = touple_edge_extractor_synset(row[target])
        
        # Create a graph using all the edge inputs as touples
        G = nx.Graph()
        G.add_edges_from(doc_edges)
        btwn_dict = nx.betweenness_centrality(G)
        Node_list = set(G.nodes())
        Node_list = list(Node_list & set(selected_synsets))
        temp_dict = {}
        for n in Node_list:
            col = str(n)+"_btw"
            temp_dict[col]= btwn_dict[n]
        synset_centrality_matrix = synset_centrality_matrix.append(temp_dict,ignore_index=True)
    
    print("Done Synset Graph Generation")
    return(synset_centrality_matrix)


###############################################################################################
################################# Summary of Feature Sets #####################################
############################################################################################### 
'''
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''

'''
doc_readability_features
word_matrix_features
synset_matrix_features
edges_matrix_features
word_centrality_matrix
edges_matrix_features_synset
synset_centrality_matrix
'''   
'''
doc_readability_features.to_excel('C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Saved_Data\\doc_readability_features.xlsx')
word_matrix_features.to_excel('C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Saved_Data\\word_matrix_features.xlsx')
synset_matrix_features.to_excel('C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Saved_Data\\synset_matrix_features.xlsx')
word_centrality_matrix.to_excel('C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Saved_Data\\word_centrality_matrix.xlsx')
synset_centrality_matrix.to_excel('C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Saved_Data\\synset_centrality_matrix.xlsx')
'''  
    