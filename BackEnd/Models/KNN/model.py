import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

clinical_df = pd.read_csv('clinical_data.csv')
stage_df = pd.read_csv('fina_Stage_unaugmented.csv', low_memory=False)

merged_df = pd.merge(clinical_df, stage_df, left_on='barcode', right_on='Samples')
merged_df = merged_df.dropna()
y = merged_df['Stage_y']
drop_cols = [
    'Unnamed: 0', 'barcode', 'Samples', 'Stage_x', 'Stage_y',
    'patient', 'sample', 'sample_submitter_id', 'sample_id'
]
X = merged_df.drop(columns=[col for col in drop_cols if col in merged_df.columns])
X = X.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#handle class imbalance of stages using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y_encoded)
# applying pca to reduce dimensionality while preseving vriance
pca = PCA(n_components=30)
X_reduced = pca.fit_transform(X_resampled)

# stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
