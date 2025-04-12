import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = './fina_Stage_unaugmented.csv'
data = pd.read_csv(file_path)
cleaned_data = data.dropna()

# Target and features
target = cleaned_data['Stage']
features = cleaned_data.drop(columns='Stage')

# Detect categorical features
categorical_columns = features.select_dtypes(include='object').columns.tolist()

# Label encode categorical columns
encoder = LabelEncoder()
for col in categorical_columns:
    features[col] = encoder.fit_transform(features[col].astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize CatBoost with automatic class weights
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='MultiClass',
    verbose=100,
    auto_class_weights='Balanced'
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Create probability DataFrame
proba_df = pd.DataFrame(y_pred_proba, columns=model.classes_)

# Ensure all expected stage columns are present
all_stages = sorted(list(set(y_train.unique()).union(set(y_test.unique()))))
for stage in all_stages:
    if stage not in proba_df.columns:
        proba_df[stage] = 0.0
proba_df = proba_df.reindex(columns=all_stages)

# Add ground truth
proba_df['True Stage'] = y_test.values

# Save per-sample probabilities
proba_df.to_csv('./result/gdb_p_result.csv', index=False)

# ---- Classification Report ----
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv('./result/classification_mtrx.csv')

print('Classification Report:\n', report_df)

# ---- AUC Score ----
try:
    y_true_bin = pd.get_dummies(y_test).reindex(columns=all_stages, fill_value=0)
    print('AUC Score:', roc_auc_score(y_true_bin, proba_df[all_stages], multi_class='ovr'))
except ValueError as e:
    print(f"Error calculating AUC Score: {e}")

# ---- Initial Probability Estimates ----
mean_probs = proba_df[all_stages].mean().sort_values(ascending=False)
mean_probs.to_csv('./result/stage_result.csv', header=['Estimated Probability'])


