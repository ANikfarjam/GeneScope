import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/content/fina_Stage_unaugmented.csv', low_memory=False)
df = df.drop_duplicates(subset='Samples')
y = df['Stage']
drop_cols = [
    'Samples', 'Stage', 'vital_status', 'submitter_id', 'barcode',
    'sample_id', 'sample', 'sample_submitter_id', 'patient', 'paper_patient',
    'diagnosis_id', 'bcr_patient_barcode',  
    'paper_age_at_initial_pathologic_diagnosis', 'paper_days_to_birth',
    'paper_pathologic_stage', 'ajcc_pathologic_n', 'ajcc_pathologic_t',
    'ajcc_pathologic_m', 'year_of_diagnosis', 'treatments', 'Unnamed: 0',
    'paper_days_to_last_followup', 'days_to_collection', 'demographic_id',
    'initial_weight', 'days_to_birth', 'pathology_report_uuid',
    'age_at_diagnosis', 'age_at_index', 'method_of_diagnosis',
    'sites_of_involvement', 'primary_diagnosis', 'morphology',
    'paper_PARADIGM.Clusters', 'paper_Mutation.Clusters', 'paper_CNV.Clusters',
    'paper_BRCA_Subtype_PAM50', 'paper_miRNA.Clusters', 'paper_DNA.Methylation.Clusters',
    'paper_Included_in_previous_marker_papers', 'paper_mRNA.Clusters',
    'ethnicity', 'preservation_method', 'race', 'laterality',
    'paper_vital_status', 'oct_embedded', 'prior_malignancy',
    'synchronous_malignancy', 'age_is_obfuscated', 'prior_treatment',
    'tissue_or_organ_of_origin', 'icd_10_code'
]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
X = X.select_dtypes(include=[np.number])

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#handle class imbalance of stages using SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)
# applying pca to reduce dimensionality while preseving vriance
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_resampled)

# stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
