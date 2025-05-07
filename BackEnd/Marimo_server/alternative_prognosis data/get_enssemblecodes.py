from Bio import Entrez
import pandas as pd
import time
import ssl
import tqdm
# Disable SSL verification (for development purposes)
ssl._create_default_https_context = ssl._create_unverified_context
# Replace with your email
Entrez.email = "ashkan.nikfarjam@sjsu.edu"
#import ahp results
ahp_df = pd.read_csv('../AHPresults/fina_Stage_unaugmented.csv')
print(ahp_df.shape)
# List of your top gene symbols
gene_symbols = ahp_df.iloc[:, -2000:].columns
with open('gene_list.txt','x') as f:
    for gene in gene_symbols[:500]:
        f.write(gene + '\n')
# results = []

# results = []

# for symbol in tqdm.tqdm(gene_symbols, total=len(gene_symbols), desc='Gennting Enssemble Ids', unit='it'):
#     try:
#         # Search for gene ID
#         search_handle = Entrez.esearch(db="gene", term=f"{symbol}[Gene Name] AND Homo sapiens[Organism]", retmode="xml")
#         search_record = Entrez.read(search_handle)
#         search_handle.close()

#         if not search_record["IdList"]:
#             results.append({"Gene Symbol": symbol, "Ensembl ID": None})
#             continue

#         gene_id = search_record["IdList"][0]

#         # Fetch gene summary from NCBI
#         fetch_handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
#         record = Entrez.read(fetch_handle)
#         fetch_handle.close()

#         # Extract Ensembl Gene ID from dbXrefs
#         ensembl_id = None
#         for ref in record[0].get("Entrezgene_xref", []):
#             db = ref["Dbtag"]["Dbtag_db"]
#             if db == "Ensembl":
#                 id_val = ref["Dbtag"]["Dbtag_tag"]["Object-id"]
#                 ensembl_id = id_val.get("Object-id_str") or str(id_val.get("Object-id_id"))
#                 break

#         results.append({"Gene Symbol": symbol, "Ensembl ID": ensembl_id})
#         time.sleep(0.35)  # NCBI rate limit

#     except Exception as e:
#         print(f"Error processing {symbol}: {e}")
#         results.append({"Gene Symbol": symbol, "Ensembl ID": None})

# # Save final results to CSV
# pd.DataFrame(results).to_csv("gene_to_ensembl_mapping.csv", index=False)
# print("Saved to gene_to_ensembl_mapping.csv")