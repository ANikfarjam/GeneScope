import pandas as pd

# Step 1: Load your datasets
model_df = pd.read_csv("model_data.csv")  # This contains Stage as a string
clinical_df = pd.read_csv("brca_metabric_clinical_data.tsv", sep="\t")  # This has stage_ordinal and numeric TNM

# Step 2: Inspect the unique values (optional but good for debugging)
print("Stage labels in model_data.csv:")
print(sorted(model_df['Stage'].dropna().unique()))

print("\nUnique stage_ordinal values in clinical data:")
print(sorted(clinical_df['stage_ordinal'].dropna().unique()))

# Step 3: Define a mapping from numerical to string-based stage labels used in model_df
stage_mapping = {
    0: "Stage 0",
    1: "Stage I",
    1.1: "Stage IA",
    1.2: "Stage IB",
    2: "Stage II",
    2.1: "Stage IIA",
    2.2: "Stage IIB",
    3: "Stage III",
    3.1: "Stage IIIA",
    3.2: "Stage IIIB",
    3.3: "Stage IIIC",
    4: "Stage IV",
    9: "Stage X"  # For unstageable
}

# Step 4: Apply the mapping
clinical_df['Stage'] = clinical_df['stage_ordinal'].map(stage_mapping)

# Drop rows where the mapping failed
clinical_df = clinical_df.dropna(subset=['Stage'])

# Step 5: Aggregate numeric TNM data per stage
agg_stage_data = clinical_df.groupby('Stage').agg({
    'tumor_size': 'mean',
    'lymph_nodes_examined_positive': 'mean'
}).reset_index().rename(columns={
    'tumor_size': 'avg_tumor_size_cm',
    'lymph_nodes_examined_positive': 'avg_lymph_nodes'
})

# Step 6: Merge the aggregated info into model_data
enriched_df = model_df.merge(agg_stage_data, on='Stage', how='left')

# Step 7: Save the enriched file
enriched_df.to_csv("model_data_with_tnm_metrics.csv", index=False)
print("✅ Enriched file saved as model_data_with_tnm_metrics.csv")
