# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:17:02 2018

@author: GTayl
"""

################################## Set-up ##########################################

# Import the required packages
import pandas as pd
import numpy as np
import itertools
import sklearn
import networkx as nx
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

# Define the list of punctuation to be removed
# define punctuation
punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''

# Read in the data from the original files

################################## Data Ingest ##########################################

doc_folder_path = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\Example_Docs"
img_dir = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Photos\\test_photos"

data = Ingest_Training_Data(doc_folder_path, img_dir)

# Or read in data from a spreadsheet
traing_data = 'C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\training_set_rel3.tsv'
data = pd.DataFrame.from_csv(traing_data, sep='\t', header=0, encoding='ISO-8859-1')

# Select any subset (Scorecard) that you want to select for training
data = data[data['essay_set']==1]

################################## Data/String Cleaning ##########################################

# Prep text
def remove_punc(text):
    text_cleaned = ""
    for char in text:
       if char not in punctuations:
           text_cleaned = text_cleaned + char
    return(text_cleaned)
    
    
def clean_words(text):
    text = str(text).lower()
    text = remove_punc(text)
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return(filtered_sentence)

def clean_sentence(text):
    string = ""
    temp = clean_words(text)
    for i in temp:
        if i ==".":
            string += i
        else:
            string += (" "+i)
    return(string)
    
    
################################## Word Feature Generation ##########################################

# Unique words in corpus
unqiue_words = set()

for index, row in data.iterrows():
    temp_words = set(clean_words(row['essay']))
    unqiue_words = unqiue_words.union(temp_words)

# Sklearn Matrix Builder
    
# Develop clean data for it to run against
data['essay_clean'] = data.apply(lambda row: clean_sentence(row['essay']),axis=1)

# develop the vocab list for the methods to search over
unique_words = list(unqiue_words)

# Train and run the sklearn vectorizor to get a sparse matrix representation
count_vec = sklearn.feature_extraction.text.CountVectorizer(vocabulary=unique_words,lowercase=False)
count_matrix = count_vec.fit_transform(data['essay_clean'])

# Convert the sparse SciPy matrix into a dense matrix and convert intoa a pandas dataframe for further analysis
count_matrix = pd.DataFrame(count_matrix.todense(),columns=unique_words)
term_frequency_counts = pd.DataFrame(count_matrix.apply(lambda column: sum(column)),columns=['Counts'])

# Define the minimum threshold to observe a item (word or synset) for it to be included in our analysis
limit = round(0.01*len(count_matrix))
selected_features = term_frequency_counts[term_frequency_counts['Counts']>limit]
selected_features = selected_features.index
word_matrix_features = count_matrix[selected_features]


################################## Synset Feature Generation ##########################################

# Convert words into their hypernym code paths 
def WordNet_Translator(term):
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

# Return ths Hypernym code (as a string) for each hypernym in the path
def return_hypernym_codes(hypernym_list):
    hypernym_codes = []
    for hypernym in hypernym_list:
        hypernym_codes.append(str(hypernym.name()).replace(".","")) #removing periods for now
    return(hypernym_codes)
        
# Unique synsets in corpus
unique_synsets = set() 

for index, row in data.iterrows():
    temp_words = set(clean_words(row['essay']))
    for word in temp_words:
        temp_synsets = WordNet_Translator(word)
        if len(temp_synsets)>0:
            unique_synsets = unique_synsets.union(temp_synsets)
        else:
            pass

# Building a Synset Translation for each essay
punctuations = '''!()-[]{};.:'"\,<>/?@#$%^&*_~'''

def synset_translated_sentence(text):
    string = ""
    temp = clean_words(text)
    for i in temp:
        i = WordNet_Translator(i)
        for s in i:
            #s.replace(".","")
            string += (" "+s)
    return(string)

# Building a Synset Translation for each essay
data['essay_synset_translation'] = data.apply(lambda row: synset_translated_sentence(row['essay']),axis=1)

# develop the vocab list for the methods to search over
unique_synsets = list(unique_synsets)

# Train and run the sklearn vectorizor to get a sparse matrix representation
count_vec_syn = sklearn.feature_extraction.text.CountVectorizer(vocabulary=unique_synsets,lowercase=False,)
count_matrix_syn = count_vec_syn.fit_transform(data['essay_synset_translation'])

# Convert the sparse SciPy matrix into a dense matrix and convert intoa a pandas dataframe for further analysis
count_matrix_syn = pd.DataFrame(count_matrix_syn.todense(),columns=unique_synsets)
synset_frequency_counts = pd.DataFrame(count_matrix_syn.apply(lambda column: sum(column)),columns=['Counts'])

# Define the minimum threshold to observe a item (word or synset) for it to be included in our analysis
limit = round(0.01*len(count_matrix_syn))
selected_features_syn = synset_frequency_counts[synset_frequency_counts['Counts']>limit]
selected_features_syn = selected_features_syn.index
synset_matrix_features = count_matrix_syn[selected_features_syn]


################################## Word - Edge Feature Generation ##########################################

sent_tokenize = nltk.tokenize.punkt.PunktSentenceTokenizer()

# Break document into sentences, then word tokens, return all combinations of tokens
def pd_edge_extractor(text):
    doc_edge_list = pd.DataFrame(columns=['Item_A','Item_B'])
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(clean_words(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list = doc_edge_list.append(pd.DataFrame(list(edge_list),columns=['Item_A','Item_B']), ignore_index=True)
    return(doc_edge_list)

def touple_edge_extractor(text):
    doc_edge_list = []
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(clean_words(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list.extend(edge_list)
        #print(len(doc_edge_list))
    return(doc_edge_list)

# Get the edge counts for all sentences in all documents
master_edge_list = pd.DataFrame(columns=['Item_A','Item_B'])
for index, row in data.iterrows():
    doc_edges = pd_edge_extractor(row['essay'])
    master_edge_list = master_edge_list.append(doc_edges, ignore_index=True)

# Calculate which edges are the most common among the documents to collect counts for
master_edge_list['edge_id'] = master_edge_list.Item_A.str.cat(master_edge_list.Item_B)
selected_edge_list = pd.DataFrame(master_edge_list['edge_id'].value_counts())
selected_edge_list = selected_edge_list[selected_edge_list['edge_id']>limit]
selected_edge_list = list(selected_edge_list.index)

# Translate document text into edges representation
def edge_translation(text):
    doc_translation = ""
    doc_touples = touple_edge_extractor(text)
    for t in doc_touples:
        temp_t = str(t[0])+str(t[1])
        doc_translation = doc_translation +" "+temp_t
    return(doc_translation)

data['edge_translation']=data.apply(lambda row: edge_translation(row['essay']), axis=1)

# Train and run the sklearn vectorizor to get a sparse matrix representation
count_vec_edges = sklearn.feature_extraction.text.CountVectorizer(vocabulary=selected_edge_list, lowercase=False)
count_matrix_edges = count_vec_edges.fit_transform(data['edge_translation'])

# Convert the sparse SciPy matrix into a dense matrix and convert intoa a pandas dataframe for further analysis
edges_matrix_features = pd.DataFrame(count_matrix_edges.todense(), columns=selected_edge_list)


################################## Word - Document Network Creation ##########################################        

# Term/Word Betweeness Centrality

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
    Node_list = list(Node_list & set(selected_features))
    temp_dict = {}
    for n in Node_list:
        col = str(n)+"_btw"
        temp_dict[col]= btwn_dict[n]
    word_centrality_matrix = word_centrality_matrix.append(temp_dict,ignore_index=True)
        
    
################################## Synset - Edge Feature Generation ##########################################

#
def synset_translated_sentence(text):
    synset_list = []
    temp = clean_words(text)
    for i in temp:
        i = WordNet_Translator(i)
        for s in i:
            synset_list.append(s)
    return(synset_list)

#
def pd_edge_extractor_synset(text):
    doc_edge_list = pd.DataFrame(columns=['Synset_A','Synset_B'])
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(synset_translated_sentence(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list = doc_edge_list.append(pd.DataFrame(list(edge_list),columns=['Synset_A','Synset_B']), ignore_index=True)
    return(doc_edge_list)

#
def touple_edge_extractor_synset(text):
    doc_edge_list = []
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(synset_translated_sentence(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list.extend(edge_list)
        #print(len(doc_edge_list))
    return(doc_edge_list)

# Get the synset edge counts for all sentences in all documents
master_edge_list_synset = pd.DataFrame(columns=['Synset_A','Synset_B'])
for index, row in data.iterrows():
    doc_edges_synset = pd_edge_extractor_synset(row['essay'])
    master_edge_list_synset = master_edge_list_synset.append(doc_edges_synset, ignore_index=True)

# Calculate which synset edges are the most common among the documents to collect counts for
master_edge_list_synset['edge_id'] = master_edge_list_synset.Synset_A.str.cat(master_edge_list_synset.Synset_B)
selected_edge_list_synset = pd.DataFrame(master_edge_list_synset['edge_id'].value_counts())
selected_edge_list_synset = selected_edge_list_synset[selected_edge_list_synset['edge_id']>limit]
selected_edge_list_synset = list(selected_edge_list_synset.index)
master_edge_list_synset.drop(master_edge_list_synset.index, inplace=True) #Drop the dataframe here to preserve system resources
        
# Translate document text into a synset edges representation
def edge_translation_synset(text):
    doc_translation = ""
    doc_touples = touple_edge_extractor_synset(text)
    for t in doc_touples:
        temp_t = str(t[0])+str(t[1])
        doc_translation = doc_translation +" "+temp_t
    return(doc_translation)

data['edge_translation_synset']=data.apply(lambda row: edge_translation_synset(row['essay']), axis=1)

# Train and run the sklearn vectorizor to get a sparse matrix representation
count_vec_edges_synset = sklearn.feature_extraction.text.CountVectorizer(vocabulary=selected_edge_list_synset, lowercase=False)
count_matrix_edges_synset = count_vec_edges_synset.fit_transform(data['edge_translation'])

# Convert the sparse SciPy matrix into a dense matrix and convert intoa a pandas dataframe for further analysis
edges_matrix_features_synset = pd.DataFrame(count_matrix_edges_synset.todense(), columns=selected_edge_list_synset)


################################## Synset - Document Network Creation ##########################################     

# Synset Betweeness Centrality

# Create Betweeness Centrality Dataframe
synset_centrality_matrix = pd.DataFrame()

# Populate DataFrame with betweenness centralities 

for index, row in data.iterrows():
    
    # Extract all edges from the docuemnt as touples
    doc_edges = touple_edge_extractor_synset(row['essay'])
    
    # Create a graph using all the edge inputs as touples
    G = nx.Graph()
    G.add_edges_from(doc_edges)
    btwn_dict = nx.betweenness_centrality(G)
    Node_list = set(G.nodes())
    Node_list = list(Node_list & set(selected_features))
    temp_dict = {}
    for n in Node_list:
        col = str(n)+"_btw"
        temp_dict[col]= btwn_dict[n]
    synset_centrality_matrix = word_centrality_matrix.append(temp_dict,ignore_index=True)
        

