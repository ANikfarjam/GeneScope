import pandas as pd
import plotly.express as px

# === Load CSVs ===
fusion_model_eval = pd.read_csv("fusion_model_evaluation.csv")
main_model_report = pd.read_csv("main_class_auc_val_test.csv")
knn_report = pd.read_csv("knn_report_smote_pca.csv")
nn_model_report = pd.read_csv("model_classification_report.csv")

# === Step 1: Define class labels ===
class_labels = nn_model_report.index.tolist()

# === Step 2: Vanilla NN ===
vnn_df = nn_model_report[["precision", "recall", "f1-score"]].copy()
vnn_df["Model"] = "Vanilla NN"
vnn_df["Class"] = class_labels

# === Step 3: KNN ===
knn_df = knn_report[["precision", "recall", "f1-score"]].copy()
knn_df["Model"] = "KNN"
knn_df["Class"] = class_labels

# === Step 4: Fusion (F1 computed manually) ===
fusion_precision = fusion_model_eval.query("Metric == 'Test Precision'")["Score"].values[0]
fusion_recall = fusion_model_eval.query("Metric == 'Test Sensitivity'")["Score"].values[0]
fusion_f1 = 2 * (fusion_precision * fusion_recall) / (fusion_precision + fusion_recall)

fusion_df = pd.DataFrame({
    "precision": [fusion_precision] * len(class_labels),
    "recall": [fusion_recall] * len(class_labels),
    "f1-score": [fusion_f1] * len(class_labels),
    "Model": ["Fusion"] * len(class_labels),
    "Class": class_labels
})

# === Step 5: Main Model (using Test AUC as proxy for f1-score only) ===
main_df = pd.DataFrame({
    "precision": [None] * len(main_model_report),
    "recall": [None] * len(main_model_report),
    "f1-score": main_model_report["Test AUC"],
    "Model": ["Main Model"] * len(main_model_report),
    "Class": main_model_report["Class"]
})

# === Step 6: Combine all ===
all_reports = pd.concat([vnn_df, knn_df, fusion_df, main_df], ignore_index=True)

# === Step 7: Melt to long format ===
melted = all_reports.melt(
    id_vars=["Model", "Class"],
    value_vars=["precision", "recall", "f1-score"],
    var_name="Metric",
    value_name="Score"
)

# === Step 8: Plot ===
fig = px.bar(
    melted,
    x="Class",
    y="Score",
    color="Model",
    facet_col="Metric",
    barmode="group",
    title="Per-Class Classification Summary Across All Models",
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig.update_layout(title_x=0.5, height=600)
fig.show()
