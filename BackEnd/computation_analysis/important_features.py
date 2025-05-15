import numpy as np
import pandas as pd
important_features = [
    "barcode", "ajcc_pathologic_stage","paper_pathologic_stage", "ajcc_pathologic_n", "ajcc_pathologic_t", "year_of_diagnosis",
    "treatments", "paper_days_to_last_followup", "days_to_collection", 
    "demographic_id", "initial_weight", "days_to_birth", 
    "pathology_report_uuid", "paper_age_at_initial_pathologic_diagnosis", "age_at_diagnosis", 
    "age_at_index", "paper_days_to_birth",
    "sites_of_involvement", "paper_PARADIGM.Clusters", "paper_Mutation.Clusters", 
    "paper_CNV.Clusters", "primary_diagnosis", "paper_BRCA_Subtype_PAM50", "morphology", 
    "paper_miRNA.Clusters", "ajcc_pathologic_m", "method_of_diagnosis", 
    "paper_DNA.Methylation.Clusters", "paper_Included_in_previous_marker_papers", 
    "paper_mRNA.Clusters", "ethnicity", "preservation_method", "race", "laterality", 
    "paper_vital_status", "oct_embedded", "vital_status", "prior_malignancy", 
    "synchronous_malignancy", "age_is_obfuscated", "prior_treatment", 
    "tissue_or_organ_of_origin", "icd_10_code"
]

clinical_df = pd.read_csv('./head1.csv')
print(clinical_df['ajcc_pathologic_m'].value_counts())
gene_exp_df = pd.read_csv('../../data/ModelDataSets/cancerExpressions.csv')
gene_exp_df = gene_exp_df.rename(columns={f'{gene_exp_df.columns[0]}':'Samples'})
#gene_exp_df = gene_exp_df.rename(columns={'0', 'Samples'})
print(gene_exp_df.head())
ahp_df = pd.read_csv('./AHPresults/final_Mod_ahp_scores.csv')
top2000_gene_df = ahp_df.sort_values(by='Scores', ascending=False)
top2000_genes  = top2000_gene_df['Gene'].to_list()[:2000]
gene_exp_df = gene_exp_df.loc[:,['Samples'] + top2000_genes]
important_cln_features = clinical_df.loc[:, clinical_df.columns.intersection(important_features)]
important_cln_features.rename(columns={'barcode':'Samples'}, inplace=True)

# Ensure 'Samples' column in both DataFrames are of the same type (string)
important_cln_features['Samples'] = important_cln_features['Samples'].astype(str)
gene_exp_df['Samples'] = gene_exp_df['Samples'].astype(str)

# Perform join on 'Samples' column
final_df = important_cln_features.merge(gene_exp_df, on='Samples', how='inner')
print(final_df.iloc[:,:len(important_features)].info(verbose=True, show_counts=True))
final_df.rename(columns={'ajcc_pathologic_stage':'Stage'},inplace=True)
col = final_df.pop('Stage')
final_df.insert(0, 'Stage', col)
print(final_df['ajcc_pathologic_m'].value_counts())
final_df.to_csv('./AHPresults/fina_Stage_unaugmented.csv', index=False)
