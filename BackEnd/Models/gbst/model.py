import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# Load the dataset
file_path = './patient_demographic.csv'
data = pd.read_csv(file_path)
# Drop rows with any NaN values
cleaned_data = data.dropna()

# Selecting relevant features and encoding categorical variables
features = cleaned_data[['ajcc_pathologic_t', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'paper_miRNA.Clusters', 'ethnicity', 'race', 'age_at_diagnosis']]
target = cleaned_data['Stage']

# Separate categorical and numerical features
categorical_features = ['ajcc_pathologic_t', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'paper_miRNA.Clusters', 'ethnicity', 'race']



# Encode each categorical column and save them in a new DataFrame
encoded_categorical_df = features[categorical_features].apply(encoder.fit_transform)

# Merge the encoded categorical features with the numerical features
numerical_features = features.drop(columns=categorical_features)
merged_features = pd.concat([encoded_categorical_df, numerical_features], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(merged_features, target, test_size=0.2, random_state=42)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, eval_metric='MultiClass', verbose=100)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Ensure the predicted probability columns match all possible stages
all_stages = sorted(list(set(y_train.unique()).union(set(y_test.unique()))))
proba_df = pd.DataFrame(y_pred_proba, columns=model.classes_)

# Add missing columns if any
for stage in all_stages:
    if stage not in proba_df.columns:
        proba_df[stage] = 0

# Reindex to ensure columns match expected stages
proba_df = proba_df.reindex(columns=all_stages, fill_value=0)

# Evaluate the model
print('Classification Report:\n', classification_report(y_test, y_pred))
try:
    print('AUC Score:', roc_auc_score(pd.get_dummies(y_test).reindex(columns=all_stages, fill_value=0), proba_df, multi_class='ovr'))
except ValueError as e:
    print(f"Error calculating AUC Score: {e}")

# Visualizing Prediction Probabilities
proba_df['True Stage'] = y_test.values
proba_df.to_csv('./result/gdb_p_result.csv', index=False)
plt.figure(figsize=(12, 8))
sns.boxplot(data=proba_df.melt(id_vars=['True Stage'], var_name='Predicted Stage', value_name='Probability'), x='True Stage', y='Probability', hue='Predicted Stage')
plt.title('Prediction Probabilities per True Stage')
plt.ylabel('Probability')
plt.xlabel('True Cancer Stage')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('./result.png')
plt.show()
