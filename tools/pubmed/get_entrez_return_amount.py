"""
get_entrez_return_amount.py
Mark Bieda, based on chatGPT-o3
June 16, 2025

Returns the number of abstracts for a given query. Edit the query etc here to get the numbers.
Notes: have an Entrez account and use the entrez email for that account
"""

from Bio import Entrez
raw_email = input("Enter your NCBI account email): ").strip()
Entrez.email = raw_email
raw_api = input("Enter your NCBI account api key: ").strip()
Entrez.api_key = raw_api

# query = "oncology[mesh] AND 2008:2025[pdat] AND hasabstract[text]"
#query = '(cancer[Title/Abstract] OR "breast neoplasms"[MeSH Terms]) AND (chromatin[Title/Abstract] OR epigenetics[Title/Abstract] OR histone[Title/Abstract] OR epigenetic[Title/Abstract] OR histones[Title/Abstract]) AND 2005:2025[dp] AND hasabstract[text]'
# query = '(cancer[Title/Abstract]) AND (ERBB2[Title/Abstract] OR HER2[Title/Abstract] OR EGFR[Title/Abstract]) AND 2000:2004[dp] AND hasabstract[text]' - 3067 abstracts
query = '(cancer[Title/Abstract]) AND (ERBB2[Title/Abstract] OR HER2[Title/Abstract] OR EGFR[Title/Abstract]) AND 2005:2025[dp] AND hasabstract[text]'

handle = Entrez.esearch(db="pubmed", term=query, retmax=0)  # retmax=0 → count only
result = Entrez.read(handle)
print("Total abstracts:", result["Count"])
