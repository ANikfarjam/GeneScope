import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


df = pd.read_csv('/content/fina_Stage_unaugmented.csv', low_memory=False)
df = df.drop_duplicates(subset='Samples')
df = df.dropna(subset=['Stage'])

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


# Label Encoding Standardize + PCA
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)


smote = SMOTE(random_state=42,  k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_pca, y_encoded)

num_classes = len(np.unique(y_resampled))
y_onehot = tf.keras.utils.to_categorical(y_resampled, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_onehot, test_size=0.2, random_state=42
)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")
present_labels = np.unique(y_true_classes)
present_class_names = label_encoder.classes_[present_labels].astype(str)

print(classification_report(
    y_true_classes, y_pred_classes,
    labels=present_labels,
    target_names=present_class_names
))

model.save_weights("brca_stage_model_smote_pca.weights.h5")
print("Weights saved!")
