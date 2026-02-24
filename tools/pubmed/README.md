# UTILITIES FOR PUBMED ABSTRACT DOWNLOADS  

## NOTE: This is barely more than a stub of a document. The programs are simple, check them to understand better.  
 
## WHAT  
These are a series of utilities for downloading pubmed abstracts.  

## FILE DESCRIPTION  
README.md: this file  
get_entrez_return_amount.py: for a given pubmed query, gets the number of abstracts produced  
separate_training_test.py: separate text into training and test sets  
text2vocab.py: get an output of the word counts from a text  
get_abstract_counts_by_year_adj.py: from a downloaded set of abstracts, get the count by year  
get_pubmed_abstracts_v7.py: the main engine. This downloads the abstracts into files  
text2tokenvalues.py: takes text and converts to token values   

## GENERAL USAGE  
1. adjust parameters in get_entrez_return_amount.py; run program to get an idea of how many abstracts will be downloaded using the parameters  
2. adjust parameters in get_pubmed_abstracts_v7.py; run the program to download the abstracts  
3. adjust parameters in separate_training_test.py; run the program to create the training, test and validation sets that are ready for domain-adaptive pretraining (DAPT)  


## IMPORTANT REQUIREMENT  
You must set up a NCBI account. This is very fast. This allows the bulk download of abstracts and the counts.  



