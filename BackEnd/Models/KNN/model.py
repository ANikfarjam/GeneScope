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

# Load and prepare dataset
df = pd.read_csv('/content/fina_Stage_unaugmented.csv', low_memory=False)
df = df.drop_duplicates(subset='Samples')
df = df.dropna(subset=['Stage'])

y = df['Stage']
drop_cols = [
    'site_of_resection_or_biopsy', 'tumor_descriptor', 'sample_type_id', 'definition', 'primary_site',
    'name', 'disease_type', 'shortLetterCode', 'sample_type', 'project_id', 'classification_of_tumor',
    'specimen_type', 'state', 'is_ffpe', 'tissue_type', 'composition', 'paper_Tumor.Type', 'gender',
    'days_to_diagnosis', 'releasable', 'diagnosis_is_primary_disease', 'released'
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
#smote due to some stages lacking samples
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

#to reduce the dimentiality
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

train_acc = knn.score(X_train, y_train)
test_acc = knn.score(X_test, y_test)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
conf_matrix = confusion_matrix(y_test, y_pred)
conf_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)

y_true = y_test
y_pred_classes = y_pred
y_true_classes = y_test

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")

print("Classification Report:\n")
present_labels = np.unique(y_true_classes)
present_class_names = label_encoder.classes_[present_labels].astype(str)

print(classification_report(
    y_true_classes, y_pred_classes,
    labels=present_labels,
    target_names=present_class_names
))


report_df.to_csv("knn_report_smote_pca.csv", index=True)
conf_df.to_csv("knn_confusion_matrix_smote_pca.csv", index=True)
print("Report and confusion matrix saved.")
