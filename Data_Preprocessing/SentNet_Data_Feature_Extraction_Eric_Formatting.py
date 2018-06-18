# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:17:02 2018

@author: GTayl
"""

###############################################################################
# Overview
###############################################################################

'''
In this file we define all the feature extraction function that are used within
SentNet. Using the functions contained within this file, a corpus can be 
transformed into the following feature sets:

doc_readability_features - Readability statistics for each document drawn from 
                            academic research
word_matrix_features - Word counts for the most common/important words within 
                        each document
synset_matrix_features - The prevelence of the most common/important synsets
                          (a WordNet lexigraphic representation of words) 
                          contained within each document
edges_matrix_features - Counts of the most common/important co-occuring terms 
                         within each document
word_centrality_matrix - The centrality/importance of key terms within each 
                          document (based on a graphical translation of that 
                          document)
edges_matrix_features_synset - Counts of the most common/important co-occuring
                                synsets within each document
synset_centrality_matrix - The centrality/importance of key synsets within each 
                            document (based on a graphical translation of 
                            that document)

'''   

###############################################################################
# Set-up 
###############################################################################

'''
Below are all the required packages and global variables that are required to 
execute the functions contained within this file
'''

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
    stop_words = stopwords.words('english')
    
except:
    print(""" Note the English NLTK corpus will need to be downloaded for these 
          to function properly. To download the corpus you must run the following:
            
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
punctuations_1 = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''
punctuations_2 = '''!()-[]{};.:'"\,<>/?@#$%^&*_~'''

# Defining the sentence tokenization method to be used later on
sent_tokenize = nltk.tokenize.punkt.PunktSentenceTokenizer()

###############################################################################
# Data Ingest
###############################################################################

################################## .Docx Data Ingest ##########################
'''
# Specify a folder that contains the training document in .docx format
doc_folder_path = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\Example_Docs"
img_dir = "C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Photos\\test_photos"

# Use the Ingest Training Data function from SentNet_Data_Prep_Functions.py 
#   to read in the training data 
data = Ingest_Training_Data(doc_folder_path, img_dir)

############################### Spreadsheet Data Ingest #######################
# Alternatively read in data from a prepoulated spreadsheet (or database table)
traing_data = 'C:\\Users\\GTayl\\Desktop\\Visionist\\SentNet\\Data\\training_set_rel3.tsv'
data = pd.DataFrame.from_csv(traing_data, sep='\t', header=0, encoding='ISO-8859-1')

# Select any subset (Scorecard) that you want to use for training
data = data[data['essay_set']==1]
print("Done Data Injest")
time1 = time.time()
print(time1-start)
'''
###############################################################################
# Data Cleaning 
###############################################################################
'''
In this section we define functions that will be used to clean/transform 
documents so they can be ingested by the later feature generation functions.
'''

# Prep text
def remove_punc(text):
    '''
    Purpose: To remove punctuation marks/symbols from the provided string. 
    Punctuation symbols to be removed are defined in the above set-up section.
        
    Input: String
    Output: String with puctuation symbols removed
    
    '''
    text_cleaned = ""
    
    for char in text:
       if char not in punctuations_1:
           text_cleaned = text_cleaned + char
           
    return(text_cleaned)
    
    
def clean_words(text):
    '''
    Purpose: To "clean" the provided sentence by removing punctuation and stop 
    words. Punctuation symbols and stop words to be removed are defined in 
    the above set-up section.
        
    Input: String
    Output: List of words (in order) with puctuation symbols and stop words 
        removed.
    
    '''
    text = str(text).lower()
    text = remove_punc(text)
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    
    return(filtered_sentence)

def clean_sentence(text):
    '''
    Purpose: To provide a cleaned/transformed string with punctuation marks 
        and stop words removed.
        
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
 
###############################################################################
# Document Features & Readability Tests
###############################################################################
'''
In this section, we define the function used to run and extract various 
"readability" metrics for every document. These features will be the first 
input into our classification models.
'''

# Create a copy of our original data pull to append to
def Readability_Features(data, target):
    
    '''
    Purpose:  In this function we define a method to calculate standardized 
                "readability" metrics for each document. These readability 
                measures are drawn from commonly accepted academic studies and 
                are calculated using existing python packages.
        
    Input: This function requires the following inputs:
        
            1) data = A pandas dataframe with the original document text 
                        included in a single column
            2) target = The "target" column within that dataframe that contains 
                        the original text for each document
        
    Output: doc_readability_features - a pandas dataframe that contains 
              readability metriccs from the Textatistic and Textstate Python
              libraries
        
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

  
###############################################################################
# Word Feature Extraction
###############################################################################
'''
In this section, we define the function to identify and extract all word/token 
based features from every document. These features will be the second input 
into our classification models.
'''

def Word_Features(data, target, limit):
    '''
    Purpose: In this function we first clean a document (removing punctuation, 
             stopword, etc). Next, we extract the unique words from all 
             documents. We then select the subset of terms that occur more than 
             the user defined threashold. Next we obtain a feature vector for 
             each term that holds the number of times that term was observed in 
             each document. These feature vectors are then appended into a 
             pandas DataFrame for further analysis.
    
    Input: The following inputs are required for this function:
        
            1) data = Pandas DataFrame with the document text stored in a 
                        column (in this case 'essay')
            2) target = The "target" column within that dataframe that contains 
                        the original text for each document
            3) limit = This parameter can be used to exclude terms that are 
                         sufficiently rare that they do not add any predictive 
                         value. A limit parameter of 0.01 means that a word must 
                         appear in at least 1% of documents to be included in 
                         the term matrix. This helps to remove extreamly rare 
                         features (as well as mispellings) that will provide 
                         little value during the modeling process.
        
    Output: A dense document (n) by word (w) (n x w) matrix that contains the 
                frequency counts of selected words in each document
    '''
    
    # Obtains a set of unique words in our corpus
    unique_words = set()
    
    for index, row in data.iterrows():
        temp_words = set(clean_words(row[target]))
        unique_words = unique_words.union(temp_words)
    
    # Sklearn Matrix Builder
        
    # Append a new column to our original dataframe with a "cleaned" version of the document text
    data['essay_clean'] = data.apply(lambda row: clean_sentence(row[target]),axis=1)
    
    # Define the vocab list for the Sklearn vecotrizer to search over
    unique_words = list(unique_words)
    
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

###############################################################################
# Synset Feature Generation
###############################################################################
'''
In the below section, we define the functions to identify and extract all 
sysnset based features from every document. These features will be the third 
input into our classification models.
'''

########################## Synset Translation Functions #######################

# Return the Hypernym code (as a string) for each hypernym in the path
def return_hypernym_codes(hypernym_list):
    '''
    Purpose: This function extracts the sysnet name/code from wn.synset object
                class. It then "cleans" this code by removing the token 
                separators allowing for easier analysis later on.
        
    Input: a list of sysnet.objects (extracted using the WordNet_Translator 
            function)
        
    Output: a list of cleaned synset codes
    '''
    
    hypernym_codes = []
    
    for hypernym in hypernym_list:
        hypernym_codes.append(str(hypernym.name()).replace(".","")) #removing periods for now
        
    return(hypernym_codes)


# Convert words into their hypernym code paths 
def WordNet_Translator(term):
    '''
    Purpose: This function "translates" a term into its synset equivalent.
             It does this by using the WordNet Lexical Database 
                 (developed at Princeton) implimneted in NLTK.
             Specifically, this function extracts the "hypernym path" for the 
                 provided term and then replaces this term with its hypernym 
                 equivlents in the document.
             Hypernyms are high level representations of that term or object.
             For example, the hypernym path for "Official" would be:
                 ['entityn01', 'physical_entityn01', 'causal_agentn01', 
                  'personn01', 'workern01', 'skilled_workern01', 
                  'officialn01']
        
    Input: A lower cased string that represents the word you would like to 
                translate (note it may not be able to translate obscure names, 
                words, or terms in a foreign language)
        
    Output: A list of sysnset codes that are contained within the provided 
                term's hypernym path (with period deliniators removed for 
                easier analysis later on)
    '''
    hypernym_list = []
    synset = wn.synsets(term)
    
    if len(synset)==0:
        pass
    
    else:
        synset = synset[0]
        hypernym_list = synset.hypernym_paths()
        while type(hypernym_list[0])==list:
            hypernym_list = hypernym_list[0]
        hypernym_list = return_hypernym_codes(hypernym_list)
        
    return(hypernym_list)


# Building a Synset Translation for each essay
def synset_translated_string(text):
    '''
    Purpose: This function takes a sentence and returns a synset translated 
                sentence in which all terms have been replaced with all the 
                synset codes along their hypernym path.
        
    Input: Sentence represented as a string
        
    Output: A string in which all terms have been replaced with all the 
                synset codes along thier hypernym path, separated by spaces.
        
    '''
    string = ""
    temp = clean_words(text)
    
    for i in temp:
        i = WordNet_Translator(i)
        for s in i:
            #s.replace(".","")
            string += (" "+s)
            
    return(string)


########################## Synset Translation & Extraction ####################
def Synset_Features(data, target, limit): 
    '''
    Purpose: In this section we extract synset counts from documents to use as 
                the third feature class for our models.
    
    Input: Pandas DataFrame with the document text stored in a column 
            (in this case 'essay')
        
    Output: A dense document (n) by synset (s) matrix (n x s) that contains 
                frequency counts of "common synsets" in each document
            Selecting what constitutes a "common" synset can be specified 
                using the "limit" parameter below. 
            A limit parameter of 0.01 means that a synset must appear in at 
                least 1% of documents to be included in the synset matrix.
            This helps to remove extreamly rare features that will provide 
                little value during the modeling process.
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

###############################################################################
# Word Graph Feature Generation
###############################################################################
'''
In this section, we define the functions to create a word network for every 
document. All terms that occur within the same sentence are considered to be 
"connected" and are represented as an edge/connection within the given 
document's network. Translating document into a network based form allows us to 
extract three additional feature sets:

1) Edge Counts - First we calculate the occurance of every pair of co-occuring 
    words (within the same sentence) across all documents. Next we exclude those 
    co-occuring words that are rare and not common enough to be a reliable 
    feature to model on (this limiting parameter can be tuned by the user).
    Finally, we look across all document to calculate the presence/abcesce of 
    selected co-occuring terms. This final represention is stored as a Pandas 
    dataframe allowing these features to be easily used for modeling.

2) Term Betweeness Centrality - For every document we will be able to extract 
    the "centrality" or importance of every term within that document. This is 
    done by first identifying the shortest path between all terms in the network.
    Next we calculate the proportion of shortest paths that traverse each term 
    (node). This proportion (bounded between 0 and 1) is refered to as the 
    "betweeness centrality" of that synset. This calculation is performed for 
    every word in every document to ascertain that term's importance within that 
    document.
                            
3) Document Clusters - After translating all documents into their network 
    representation, we aggreate all networks from all documents into a 
    single network that represents the entire corpus. From this network, we run 
    the Louvain clsutering algorithim to identify "clusters" of words within 
    all documents. The clusters that emerge represent common topics/arguements 
    found across all essays. After identifying these clusters we calculate the 
    concentration/presence of each cluster within each document.
        
These two sets of features will be used as the fourth, fifth, and sixth inputs 
    into our classification models.
'''

######################### Word - Edge Feature Generation Functions ############

# Break document into sentences, then word tokens, return all combinations of tokens
def pd_edge_extractor(text):
    '''
    Purpose: This function builds an edge/connections list for the document
        provided. Using this edge list a document graph can be created and term 
        centrality/cluster features can be inferred. This is done through the 
        following steps:
         1) Initalize a blank dataframe to store the document edge list
         2) Break a document into sentences using the Punkt Sentence Tokenizer 
            (defined previously)
         3) Clean the sentence, lowercasing all words and removing punctuation
         4) Within each sentence, all combinations of 2 words are defined and 
            appended to the dataframe
         5) After performing this edge generation opertion on all sentences, 
            the finalized document edge dataframe is returned
        
    Input: String representing the document text you would like to analyze
        
    Output: A Pandas dataframe containing all the word edges found within that 
        document
        
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
    Purpose: This function builds an edge/connections list for the documnet 
        provided. Using this edge list a document graph can be created and term 
        centrality/cluster features can be inferred. This is done through the 
        following steps:
                 
         1) Initalize a blank list to store the document edge list
         2) Break a document into sentences using the Punkt Sentence Tokenizer 
            (defined previously)
         3) Clean the sentence, lowercasing all words and removing punctuation
         4) Within each sentence, all combinations of 2 words are defined and 
            appended to the list (terms are ordered alphabetically before 
            constructing the touple)
         5) After performing this edge generation opertion on all sentences, 
            the finalized document edge list is returned
        
    Input: String representing the document text you would like to analyze
        
    Output: A list containing all the word edges infered from that document
    '''
    
    doc_edge_list = []
    sent_list = sent_tokenize.tokenize(text)
    
    for sent in sent_list:
        term_list = sorted(clean_words(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list.extend(edge_list)

    return(doc_edge_list)

# Translate document text into edges representation
def edge_translation(text):
    '''
    Purpose: This function translates a document into a representation of all 
    of its edges. This is done through the following steps:
                 
     1) Initalize a blank string to hold the translated doucment
     2) Extract all edges (represented as touples) using the 
        touple_edge_extractor function
     3) Concatenate the touple into a single string which we will use as the 
        edge ID
     4) Append this new edge ID to the translated document string
        
    Input: String representing the document that you would like to translate
        
    Output: String that contains all edge id's (concatenated string containing 
            str(Item_A)+str(Item_B)) contained within that document
    '''
    
    doc_translation = ""
    doc_touples = touple_edge_extractor(text)
    
    for t in doc_touples:
        temp_t = str(t[0])+str(t[1])
        doc_translation = doc_translation +" "+temp_t
        
    return(doc_translation)

######################### Word - Edge Feature Extraction ######################
def Word_Edge_Features(data, target, limit):
    '''
    Purpose: In this function we build a word network for all documents, 
    find common co-occuring terms, then calculate the prevelence of those 
    co-occuring terms across all documents. This co-occurence matrix is then 
    used as an input to our modeling feature space.
        
    Input: This function requires the following inputs:
        1) data = A pandas dataframe the contains the documents that you are 
            interested in analyzing
        2) target = The column/feature within the dataframe the contains 
            the original document text
        3) limit = This parameter can be used to exclude terms that are 
            sufficiently rare that they do not add any predictive value. 
            A limit parameter of 0.01 means that a co-occuring term pair must 
            appear in at least 1% of documents to be included in the term matrix. 
            This helps to remove extreamly rare features (as well as mispellings) 
            that will provide little value during the modeling process. 
        
    Output: This function returns a document (d) by feature (f) (d x f) Pandas 
        DataFrame that contains a count of the number of times a co-occuring 
        term pair is found within each document.
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


#################### Word - Betweeness Centrality Extraction ##################       

def Word_Centrality_Features(data, target, selected_words):
    '''
    Purpose: This function returns the "importance" of every word, within every 
        document through the use of graph analysis through the following steps:
        1) A blank pandas DataFrame (word_centrality_matrix) is initalized to 
            hold the resulting word betweeness centralities in every document
        2) All documents are translated into a co-occuring term/edge 
            representation using the touple_edge_extractor function
        3) A blank graph is initalized for each document using NetworkX
        4) All co-occuring terms/edges are fed into a NetworkX graph
        5) For "selected terms" (defined in the Word_Features function) the 
            betweenness centrality is calculated for all terms then 
            appeneded to the word_centrality_matrix
        6) Once this is complete for all terms in all documents the finalized 
            DataFrame is returned
        
    Input: This function requires the following inputs:
        
            1) data = a pandas DataFrame that contains the documents for 
                processing
            2) target = the column/feature within the DataFrame that contains 
                the original document text
            3) selected_words = a list of words to calculate betweeness 
                centrality scores for (note this list can be returned by 
                Word_Features function).
        
    Output: A pandas DataFrame that contains the betweeness centrality of 
        selected terms within all documents
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
        
        # Calculate the betweeness centrality for all terms on the Node_list within graph G
        for n in Node_list:
            col = str(n)+"_btw"
            temp_dict[col]= btwn_dict[n]
            
        word_centrality_matrix = word_centrality_matrix.append(temp_dict,ignore_index=True)

    # Return the final word_centrality DataFrame
    print("Done Word Graph Generation")
    return(word_centrality_matrix)

        
###############################################################################
# Synset Graph Feature Generation
###############################################################################
'''
In this section, we define the functions to create a synset network for every 
document. Synsets that occur within the same sentence share an edge within 
the network. Translating document into a network based form allows us to 
extract two additional feature sets:
    
    1) Synset Edge Counts - First we calculate the occurance of every pair of 
        co-occuring synsets (within the same sentence) across all documents.
        Next we exclude those co-occuring sysents that are rare and not 
        common enough to be a reliable feature to model on (this limiting 
        parameter can be tuned by the user). Finally, we look across all 
        document to calculate the presence/abcesce of selected co-occuring 
        synsets. This final represention is stored as a Pandas dataframe 
        allowing these features to be easily used for modeling.
    
    2) Synset Betweeness Centrality - For every document we will be able to 
        extract the "centrality" or importance of every synset within that 
        document. This is done by first identifying the shortest path between 
        all synset in the network. Next we calculate the proportion of shortest 
        paths that traverse each synset. This proportion (bounded between 
        0 and 1) is refered to as the "betweeness centrality" of that synset.
        This calculation is performed for every synset in every document 
        to ascertain that synset's importance within that document.
                                
    3) Synset Document Clusters - After translating all documents into their 
        synset network representation, we aggreate all networks from all 
        documents into a single network that represents the entire corpus. From 
        this network, we run the Louvain clsutering algorithim to identify 
        "clusters" of synsets within all documents. The clusters that emerge 
        represent common topics/arguements found across all essays. After 
        identifying these clusters, we calculate the concentration/presence 
        of each cluster within each document.
        
These two sets of features will then be used as the sixth and seventh inputs 
into our classification models.
'''
   
######################### Synset - Edge Feature Generation Functions ##########
#
def synset_translated_sentence(text):
    '''
    Purpose: This function takes a given string and returns a list of all the 
        synsets for all the terms contained within that string. This is done 
        through the following:
         1) A blank synset list is initalized
         2) The string is cleaned, lowercasing all text as well as removing 
            stopwords and punctuation
         3) Every term contained within that string is translated into it
            WordNet equivlent, and the hypernym synset path for that term is 
            appended to the synset list
         4) Once this transformation is performed for all words in all documents 
            a finalized synset_list is returned
        
    Input: A string contining the text you would like to translate
        
    Output: a list of all the synset (has well as higher order synsets for 
        each term) are returned
        
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
    Purpose: This function builds a list of all the synset edge/connections 
        contained within the provided documnet. Using this edge list, a 
        document graph can be created and synset centrality/cluster features 
        can be inferred. This is done through the following steps:
         1) Initalize a blank dataframe to store the document edge list
         2) Break a document into sentences using the Punkt Sentence Tokenizer 
            (defined previously)
         3) Clean the sentence, lowercasing all words and removing punctuation
         4) Translate each term within the sentence into it's synset components 
            (all synset's contained on it hypernym path) using the 
            synset_translated_sentence function
         5) Within each sentence, all combinations of 2 synsets are found and 
            then appended to the dataframe
         5) After performing this edge generation opertion on all sentences, 
            the finalized document edge dataframe is returned

    Input: String representing the document text you would like to analyze
        
    Output: A Pandas dataframe containing all the synset edges found within that
        document
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
    Purpose: This function builds an edge/connections list for the document 
        provided. Using this edge list a document graph can be created and 
        term centrality/cluster features can be inferred. This is done through 
        the following steps:
         1) Initalize a blank list to store the document edge list
         2) Break a document into sentences using the Punkt Sentence Tokenizer 
            (defined previously)
         3) Clean the sentence, lowercasing all words and removing punctuation
         4) Within each sentence, all combinations of 2 words are defined and 
            appended to the list (terms are ordered alphabetically before 
            constructing the touple)
         5) After performing this edge generation opertion on all sentences, 
            the finalized document edge list is returned
        
    Input: String representing the document text you would like to analyze
        
    Output: A list containing all the word edges infered from that document
    '''
    
    doc_edge_list = []
    sent_list = sent_tokenize.tokenize(text)
    
    for sent in sent_list:
        term_list = sorted(synset_translated_sentence(sent))
        edge_list = list(itertools.combinations(term_list, 2))
        doc_edge_list.extend(edge_list)

    return(doc_edge_list)

# Translate document text into a synset edges representation
def edge_translation_synset(text):
    '''
    Purpose: This function translates a document into a representation of all 
        of its synset edges. This is done through the following steps:
         1) Initalize a blank string to hold the translated doucment
         2) Extract all synset edges (represented as touples) using the 
            touple_edge_extractor function
         3) Concatenate each touple into a single string which we will use as 
            the edge ID
         4) Append this new edge ID to the translated document string
        
    Input: String representing the document that you would like to translate
        
    Output: A nwe string that contains all edge id's (concatenated string 
        containing str(Synset_A)+str(Synset_B)) contained within that document
    '''
    
    doc_translation = ""
    doc_touples = touple_edge_extractor_synset(text)
    
    for t in doc_touples:
        temp_t = str(t[0])+str(t[1])
        doc_translation = doc_translation +" "+temp_t
        
    return(doc_translation)

######################### Synset - Edge Feature Extraction ####################
def Synset_Edge_Features(data, target, limit):
    '''
    Purpose: In this function we build a synset network for all documents, find 
        common co-occuring synsets, then calculate the prevelence of those 
        co-occuring synsets across all documents. This co-occurence matrix is 
        then used as an input to our modeling feature space.
        
    Input: This function requires the following inputs:
        1) data = A pandas dataframe the contains the documents that you are 
            interested in analyzing
        2) target = The column/feature within the dataframe the contains the 
            original document text
        3) limit = This parameter can be used to exclude synsets that are 
            sufficiently rare that they do not add any predictive value. A limit 
            parameter of 0.01 means that a co-occuring synset pair must appear 
            in at least 1% of documents to be included in the term matrix. This 
            helps to remove extreamly rare features (as well as mispellings) 
            that will provide little value during the modeling process. 
        
    Output: This function returns a document (d) by feature (f) (d x f) Pandas 
        DataFrame that contains a count of the number of times a co-occuring 
        synset pair is found within each document.
        
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

#################### Synset - Betweeness Centrality Extraction ################

def Synset_Centrality_Features(data, target, selected_synsets):
    '''
    Purpose: This function returns the "importance" of every synset, within 
        every document through the use of graph analysis through the 
        following steps:
        1) A blank pandas DataFrame (word_centrality_matrix) is initalized to 
            hold the resulting synset betweeness centralities in every document
        2) All documents are translated into a co-occuring synset/edge 
            representation using the touple_edge_extractor_synset function
        3) A blank graph is initalized for each document using NetworkX
        4) All co-occuring synset/edges are fed into a NetworkX graph
        5) For "selected synset" (defined in the Synset_Features function) the 
            betweenness centrality is calculated for all synset then appeneded 
            to the synset_centrality_matrix
        6) Once this is complete for all synset in all documents the finalized 
            DataFrame is returned
        
    Input: This function requires the following inputs:
            1) data = a pandas DataFrame that contains the documents for processing
            2) target = the column/feature within the DataFrame that contains 
                the original document text
            3) selected_synset = a list of synset to calculate betweeness 
                centrality scores for (note this list is returned by 
                synset_Features function).
        
    Output: A pandas DataFrame that contains the betweeness centrality of 
        selected synset within all documents
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
        
        # Calcuate the betweeness centralities of all synsets contained within 
        #   the Node_List
        for n in Node_list:
            col = str(n)+"_btw"
            temp_dict[col]= btwn_dict[n]
        synset_centrality_matrix = synset_centrality_matrix.append(
                temp_dict,ignore_index=True)
    
    print("Done Synset Graph Generation")
    return(synset_centrality_matrix)


###############################################################################
# Exporting of Feature Sets 
###############################################################################

# Exporting Feature Sets - If desired, use the below functions to export all 
#                           finalized feature sets for further analysis 
'''
doc_readability_features.to_excel('doc_readability_features.xlsx')
word_matrix_features.to_excel('C:word_matrix_features.xlsx')
synset_matrix_features.to_excel('synset_matrix_features.xlsx')
word_centrality_matrix.to_excel('word_centrality_matrix.xlsx')
synset_centrality_matrix.to_excel('synset_centrality_matrix.xlsx')
''' 
    