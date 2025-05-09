from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
# Load dataset
stagedata = pd.read_csv('../AHPresults/fina_Stage_unaugmented.csv')

# Preprocessing
clinical_df = stagedata.copy().iloc[:, :-2000]
clinical_df.dropna(inplace=True)
target = 'Stage'
features = clinical_df.columns[1:]

# Label encode categorical features
le = LabelEncoder()
for col in features:
    if clinical_df[col].dtype == 'object':
        clinical_df[col] = le.fit_transform(clinical_df[col].astype(str))

X = clinical_df[features].copy()
y = clinical_df[target]
X.fillna(X.mean(), inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]
non_zero_indices = [i for i in indices if feature_importances[i] != 0]

feature_ranking_df = pd.DataFrame({
    'Feature': [features[i] for i in non_zero_indices],
    'Importance': [feature_importances[i] for i in non_zero_indices]
})

feature_ranking_df = feature_ranking_df.sort_values(by='Importance', ascending=False)
feature_ranking_df.reset_index(drop=True, inplace=True)
feature_ranking_df.to_csv('rd.csv', index=False)