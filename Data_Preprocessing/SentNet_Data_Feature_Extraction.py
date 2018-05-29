# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:17:02 2018

@author: GTayl
"""
################################## Set-up ##########################################

# Import the required packages
import pandas as pd
import itertools
import sklearn
from SentNet_Data_Prep_Functions import Ingest_Training_Data
from Goldberg_Perceptual_Hashing import ImageSignature
gis = ImageSignature()

# Note the English NLTK corpus will need to be downloaded for these to function properly
# To download the corpus you must run the following:
import nltk
# nltk.download()
# Select the following items to download:
# 1) Stopwords
# 2) WordNet
# 3) WordNet_IC
# 4) Punkt

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
stop_words = set(stopwords.words('english'))

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


################################## Edge Feature Generation ##########################################

sent_tokenize = nltk.tokenize.punkt.PunktSentenceTokenizer()

# Break document into sentences, then word tokens, return all combinations of tokens
def edge_extractor(text):
    doc_edge_list = pd.DataFrame(columns=['Item_A','Item_B'])
    sent_list = sent_tokenize.tokenize(text)
    for sent in sent_list:
        term_list = sorted(clean_words(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list = doc_edge_list.append(pd.DataFrame(list(edge_list),columns=['Item_A','Item_B']), ignore_index=True)
    return(doc_edge_list)

# Get the edge counts for all sentences in all documents
master_edge_list = pd.DataFrame(columns=['Item_A','Item_B'])

for index, row in data.iterrows():
    master_edge_list = master_edge_list.append(edge_extractor(row['essay']), ignore_index=True)




        
        
        







# Wordnet Notes
word_1 = 'rocket'
word_2 = 'missle'

word_1 = wn.synsets('rocket')
for synset in word_1:
    print(synset.definition())

word_2 = wn.synsets('missile')
for synset in word_2:
    print(synset.definition())

word_1 = word_1[0]
word_1.hypernym_paths()

word_2 = word_2[0]
word_2.hypernym_paths()