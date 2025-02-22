from rich.progress import Progress
import pandas as pd
"""
Author: A Nikfarjam
Since we are aiming to analyze early stage of breast cancer
We need to find out what genes are expressed in DCIS
Data collected analyze stages of DCIS 
"""
#load data

with open("../../data/DCIS_Progression_data.soft","r") as file:
    data=file.readlines()

# expressions are in these lines 112 - 54725
colums_texts="ID_REF	IDENTIFIER	GSM88859	GSM88861	GSM88863	GSM88865	GSM88867	GSM88869	GSM88871	GSM88858	GSM88860	GSM88862	GSM88864	GSM88866	GSM88868	GSM88870"
columns=[colums_texts.split('\t')][0]
print(f"columns: {columns}")
extract_gene_info = lambda gene: gene.strip().split('\t')
results_dic = {
  'ID_REF': [],
  'IDENTIFIER': [],
  'GSM88859': [],
  'GSM88861': [],
  'GSM88863': [],
  'GSM88865': [],
  'GSM88867': [],
  'GSM88869': [],
  'GSM88871': [],
  'GSM88858': [],
  'GSM88860': [],
  'GSM88862': [],
  'GSM88864': [],
  'GSM88866': [],
  'GSM88868': [],
  'GSM88870': []
}
with Progress() as progress:
    task = progress.add_task("Generating Data!!!",  total=abs(112 - 54725))
    for row in data[113 : 54725]:
        cols = extract_gene_info (row)
        results_dic['ID_REF'].append(cols[0])
        results_dic['IDENTIFIER'].append(cols[1])
        results_dic['GSM88859'].append(cols[2])
        results_dic['GSM88861'].append(cols[3])
        results_dic['GSM88863'].append(cols[4])
        results_dic['GSM88865'].append(cols[5])
        results_dic['GSM88867'].append(cols[6])
        results_dic['GSM88869'].append(cols[7])
        results_dic['GSM88871'].append(cols[8])
        results_dic['GSM88858'].append(cols[9])
        results_dic['GSM88860'].append(cols[10])
        results_dic['GSM88862'].append(cols[11])
        results_dic['GSM88864'].append(cols[12])
        results_dic['GSM88866'].append(cols[13])
        results_dic['GSM88868'].append(cols[14])
        results_dic['GSM88870'].append(cols[15])
        progress.update(task, advance=1)

result_df = pd.DataFrame(results_dic, columns=columns)
print(result_df.head())
result_df.to_csv("expressed_DCIS_Genes.csv", index=False)