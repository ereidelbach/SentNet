#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
:DESCRIPTION:
     In this script we define all the functions that are required to perform 
     document clustering using both words and synsets

:REQUIRES:
    NONE
    
:TODO:
    NONE
"""

# =============================================================================
# Document Clustering Set-up
# =============================================================================
import networkx as nx
import pandas as pd

try:
    from community import best_partition
except:
    print("To import the community package you must first: " +
          "`pip install python-louvain`")
    
# import functions from other python files
from Data_Preprocessing.SentNet_Data_Feature_Extraction \
import clean_words, synset_translated_list

# =============================================================================
# Word Based Clustering
# =============================================================================

def Clustering_Features(edge_table, limit, remove_duplicates=False):
    '''
    Purpose: This function is used to develop topic clusters using words.
             It accomplishes this through the following steps:
                 
                 1) Using an existing edge table (with Item_A, Item_B, and an 
                        edge_ID) this function first counts the number of times 
                        each edge/connection occurs within our training corpus
                 2) We then subset this list to only focus on co-occurring 
                        items that occur more than the user provided limit
                 3) Next we feed the co-occuring terms that exceed this 
                        threshold into a NetworkX graph, creating a new "edge" 
                        or connection within the graph for every set of 
                        co-occuring items in our table
                 4) After this we use the Louvain algorithim to partition the 
                        word graph into clusters and assign each term within 
                        the graph to one of these clusters
        
    Input: This function requires the following inputs:
            
            1) edge_table: An edge table provided by the word_edge_features 
                    function
            2) limit: The minimum number of times a group of co-occuring 
                    items need to be observed together in order to be included 
                    in our clusters
            3) remove_duplicates: Set to "False" by default. 
                    If set to "True," an edge will only be added to the graph 
                    once which can change the clusters produced
        
    Output: This function returns a pandas DataFrame that maps terms within the
                graph to their corresponding topic cluster
    '''
    
    # Edges Group By Edge to Subset
    #edge_counts = edge_table
    #edge_counts.groupby(edge_counts.columns.tolist(), as_index=False).size()
    
    # Subset Edges by Count
    edge_subset = pd.DataFrame(edge_table.groupby('edge_id').edge_id.count())
    edge_subset = edge_subset[edge_subset['edge_id']>limit]
    selected_edges = list(edge_subset.index)
    
    # Select Edges for Graph
    edge_subset = edge_table[edge_table['edge_id'].isin(selected_edges)]
    
    if remove_duplicates==True:
        edge_subset = edge_subset.drop_duplicates(subset=['edge_id'], keep='first')
        
    edge_subset['Item_A'] = edge_subset['Item_A'].astype(str)
    edge_subset['Item_B'] = edge_subset['Item_B'].astype(str)
    
    # Create Graph
    G=nx.Graph()
    
    for index, row in edge_subset.iterrows():
        
        G.add_edge(row['Item_A'], row['Item_B'])
    
    # Define Partitions
    partition = best_partition(G)
    partition_list = pd.DataFrame(partition, index=['Cluster_ID'])
    partition_list = partition_list.transpose()
    
    return(partition_list)
    

def cluster_concentration(string, cluster_term_list):
    '''
    Purpose: This function returns the percentage of terms from a given topic 
                that are contained within the provided document
        
    Input: This function takes two inputs:
        
            1) string: the original document text
            2) cluster_term_list: a list of terms that comprise the given topic
        
    Output: a float between 0 and 1 that represent how many terms from the 
                given topic are contained within the document
    '''
    
    doc_words = clean_words(string)
    intersection = len(list(set(doc_words) & set(cluster_term_list)))
    cluster_total = len(cluster_term_list)
    cluster_concentration = intersection/cluster_total
    
    return(cluster_concentration)


def Cluster_Concentrations(data, doc, partition_list):
    '''
    Purpose: This function used the cluster_concentration to obtain the 
                "concentration" or "representation" of all word based clusters 
                within a given document
        
    Input: This function requires three inputs:
        
            1) data: pandas Dataframe that contains the documents you would 
                    like the score
            2) doc: string that references the specific column/feature within 
                    that document that contains the original document text
            3) partition_list: pandas DataFrame that maps terms to clusters 
                    within the entire corpus (provided by the 
                    Clustering_Features function)
        
    Output: a pandas DataFrame that contains a measure of cluster 
            concentration, for every cluster, in every document
    '''
    
    # For loop to extract cluster concentrations in each document
    cluster_features = pd.DataFrame()
    
    # Define the number of clusters
    cluster_list = partition_list.drop_duplicates(
            subset=['Cluster_ID'], keep='first')
    num_clusters = list(cluster_list['Cluster_ID'])
    
    for c in num_clusters:
        
        cluster_term_list = partition_list[partition_list['Cluster_ID']==c]
        cluster_term_list = list(cluster_term_list.index)
        cluster_name = "Word_Cluster_"+str(c)
        
        cluster_features[cluster_name] = data.apply(
                lambda row: cluster_concentration(
                        row[doc], cluster_term_list), axis=1)
    
    return(cluster_features)


# =============================================================================
# Synset Based Clustering
# =============================================================================

def Synset_Clustering_Features(edge_table, limit, remove_duplicates=False):
    '''
    Purpose: This function is used to develop topic clusters using synsets.
             It accomplishes this through the following steps:
                 
                 1) Using an existing edge table (with Synset_A, Synset_B, 
                      and an edge_ID) this function first counts the number of 
                      times each edge/connection occurs within our training 
                      corpus
                 2) We then subset this list to only focus on co-occurring 
                      items that occur more than the user provided limit
                 3) Next we feed the co-occuring terms that exceed this 
                      threshold into a NetworkX graph, creating a new "edge" or 
                      connection within the graph for every set of co-occuring 
                      items in our table
                 4) After this we use the Louvain algorithim to partition the 
                      work/synset graph into clusters and assign each synset 
                      within the graph to one of these clusters
        
    Input: This function requires the following inputs:
            
            1) edge_table: an edge table provided by the synset_edge_features 
                             function
            2) limit: The minimum number of times a group of co-occuring items 
                        need to be observed together in order to be included 
                        in our clusters
            3) remove_duplicates: Set to "False" by default, if set to "True" 
                        an edge will only be added to the graph once which 
                        can change the clusters produced
        
    Output: This function returns a pandas DataFrame that maps synsets within 
            the graph to thier coresponding topic cluster
    '''
    
    # Edges Group By Edge to Subset
    #edge_counts = edge_table
    #edge_counts.groupby(edge_counts.columns.tolist(), as_index=False).size()
    
    # Subset Edges by Count
    edge_subset = pd.DataFrame(edge_table.groupby('edge_id').edge_id.count())
    edge_subset = edge_subset[edge_subset['edge_id']>limit]
    selected_edges = list(edge_subset.index)
    
    # Select Edges for Graph
    edge_subset = edge_table[edge_table['edge_id'].isin(selected_edges)]
    
    if remove_duplicates==True:
        edge_subset = edge_subset.drop_duplicates(subset=['edge_id'], keep='first')
        
    edge_subset['Synset_A'] = edge_subset['Synset_A'].astype(str)
    edge_subset['Synset_B'] = edge_subset['Synset_B'].astype(str)
    
    # Create Graph
    G=nx.Graph()
    
    for index, row in edge_subset.iterrows():
        
        G.add_edge(row['Synset_A'], row['Synset_B'])
    
    # Define Partitions
    partition = best_partition(G)
    partition_list = pd.DataFrame(partition, index=['Cluster_ID'])
    partition_list = partition_list.transpose()
    
    return(partition_list)
    

def synset_cluster_concentration(string, cluster_synset_list):
    '''
    Purpose: This function returns the percentage of synsets from a given topic 
                that are contained within the provided document
        
    Input: This function takes two inputs:
        
            1) string: the original document text
            2) cluster_term_list: a list of synsets that comprise the given topic
        
    Output: a float between 0 and 1 that represent how many synsets from the 
                given topic are contained within the document
    '''
    
    doc_synsets = synset_translated_list(string)
    intersection = len(list(set(doc_synsets) & set(cluster_synset_list)))
    cluster_total = len(cluster_synset_list)
    cluster_concentration = intersection/cluster_total
    
    return(cluster_concentration)

def Synset_Concentrations(data, doc, partition_list):
    '''
    Purpose: This function used the cluster_concentration to obtain the 
                "concentration" or "representation" of all synset based 
                clusters within a given document
        
    Input: This function requires three inputs:
        
            1) data: pandas Dataframe that contains the documents you would 
                        like the score
            2) doc: string that references the specific column/feature within 
                        that document that contains the original document text
            3) partition_list: pandas DataFrame that maps synsets to clusters 
                    within the entire corpus (provided by the 
                    Synset_Clustering_Features function)
        
    Output: a pandas DataFrame that contains a measure of cluster concentration, 
            for every cluster, in every document
    '''
    
    # For loop to extract cluster concentrations in each document
    cluster_features = pd.DataFrame()
    
    # Define the number of clusters
    cluster_list = partition_list.drop_duplicates(
            subset=['Cluster_ID'], keep='first')
    num_clusters = list(cluster_list['Cluster_ID'])
    
    for c in num_clusters:
        
        cluster_synset_list = partition_list[partition_list['Cluster_ID']==c]
        cluster_synset_list = list(cluster_synset_list.index)
        cluster_name = "Synset_Cluster_"+str(c)
        
        cluster_features[cluster_name] = data.apply(
                lambda row: synset_cluster_concentration(
                        row[doc], cluster_synset_list), axis=1)
    
    return(cluster_features)
