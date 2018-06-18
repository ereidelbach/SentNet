# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 01:09:24 2018

@author: GTayl
"""

# =============================================================================
# Word Clustering 
# =============================================================================

# SentNet - Clusters/Communities

# https://blog.dominodatalab.com/social-network-analysis-with-networkx/

import networkx as nx
import community
import matplotlib.pyplot as plt

# Define the proportion of each cluster present in each document

# Function to extract cluster concentrations
def cluster_concentration(string, cluster_term_list):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    doc_words = clean_words(string)
    intersection = len(list(set(doc_words) & set(cluster_term_list)))
    cluster_total = len(cluster_term_list)
    cluster_concentration = intersection/cluster_total
    
    return(cluster_concentration)

def Clustering_Features(edge_table, limit):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # Edges Group By Edge to Subset
    #edge_counts = edge_table
    #edge_counts.groupby(edge_counts.columns.tolist(), as_index=False).size()
    
    # Subset Edges by Count
    edge_subset = pd.DataFrame(edge_table.groupby('edge_id').edge_id.count())
    edge_subset = edge_subset[edge_subset['edge_id']>limit]
    sel_edges = list(edge_subset.index)
    
    # Select Edges for Graph
    edge_subset = edge_counts[edge_counts['edge_id'].isin(l)]
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


def Cluster_Concentrations(data, partition_list):
    '''
    Purpose:
        
    Input:
        
    Output:
    '''
    
    # For loop to extract cluster concentrations in each document
    cluster_features = pd.DataFrame()
    
    # Define the number of clusters
    cluster_list = partition_list.drop_duplicates(subset=['Cluster_ID'], keep='first')
    num_clusters = list(cluster_list['Cluster_ID'])
    
    for c in num_clusters:
        
        cluster_term_list = partition_list[partition_list['Cluster_ID']==c]
        cluster_term_list = list(cluster_term_list.index)
        cluster_name = "Word_Cluster_"+str(c)
        
        cluster_features[cluster_name] = data.apply(lambda row: cluster_concentration(row['essay'], cluster_term_list), axis=1)
    
    return(cluster_features)
    
    

# =============================================================================
# Synset Clustering 
# =============================================================================

edge_counts = edges
edge_counts.groupby(edge_counts.columns.tolist(), as_index=False).size()

# Subset Edges by Count
edge_subset = pd.DataFrame(edge_counts.groupby('edge_id').edge_id.count())
edge_subset = edge_subset[edge_subset['edge_id']>20]
sel_edges = list(edge_subset.index)

# Select Edges for Graph
edge_subset = edge_counts[edge_counts['edge_id'].isin(l)]
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
cluster_list = partition_list.drop_duplicates(subset=['Cluster_ID'], keep='first')
cluster_list = list(cluster_list['Cluster_ID'])

# Define the proportion of each cluster present in each document

# Function to extract cluster concentrations
def cluster_concentration(string, cluster_term_list):
    
    doc_words = clean_words(string)
    intersection = len(list(set(doc_words) & set(cluster_term_list)))
    cluster_total = len(cluster_term_list)
    cluster_concentration = intersection/cluster_total
    
    return(cluster_concentration)

# For loop to extract cluster concentrations in each document
for c in cluster_list:
    
    cluster_term_list = partition_list[partition_list['Cluster_ID']==c]
    cluster_term_list = list(cluster_term_list.index)
    cluster_name = "Synset_Cluster_"+str(c)
    
    data[cluster_name] = data.apply(lambda row: cluster_concentration(row['essay'], cluster_term_list), axis=1)


### Create a list of terms for each parition


### Search across documents to see proportion of words for each partition in each document

# draw nodes, edges and labels
nx.draw(G)
plt.show()