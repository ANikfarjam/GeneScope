from Bio import Entrez
import ssl
import pandas as pd
import tqdm

# Disable SSL verification (for development purposes)
ssl._create_default_https_context = ssl._create_unverified_context

# Set your email - NCBI requires this
Entrez.email = "ashkan.nikfajram@sjsu,edu"

# List of gene symbols or names
datas = pd.read_csv('./AHPresults/final_Mod_ahp_scores.csv')
datas = datas.sort_values(by='Scores')
gene_list = datas['Gene'].to_list()[:500]

def fetch_gene_description(gene_name, organism="Homo sapiens"):
    try:
        # Search for the gene
        handle = Entrez.esearch(db="gene", term=f"{gene_name}[Gene Name] AND {organism}[Organism]")
        record = Entrez.read(handle)
        handle.close()
        
        if record["IdList"]:
            gene_id = record["IdList"][0]
            # Fetch the gene summary
            summary_handle = Entrez.esummary(db="gene", id=gene_id)
            summary_record = Entrez.read(summary_handle)
            summary_handle.close()
            
            description = summary_record['DocumentSummarySet']['DocumentSummary'][0]['Description']
            return description
        else:
            return f"No description found for {gene_name}"
    
    except Exception as e:
        return f"Error fetching {gene_name}: {e}"

# Fetch and print descriptions for all genes
desc_list = []
for gene in tqdm.tqdm(gene_list, desc='Fetching info from NCBI', total=len(gene_list), unit='it'):
    desc = fetch_gene_description(gene)
    # print(f"{gene}: {desc}")
    desc_list.append(desc)
result_df = pd.DataFrame({'Genes': gene_list, 'Description':desc_list})
print(result_df.head())
result_df.to_csv('./AHPresults/top500_desc.csv', index=False)
print('Done!')

