#script for MLP model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

clinical_df = pd.read_csv('/content/clinical_data.csv')
stage_df = pd.read_csv('/content/fina_Stage_unaugmented.csv')

merged_df = pd.merge(clinical_df, stage_df, left_on='barcode', right_on='Samples')
merged_df = merged_df.dropna()

y = merged_df['Stage_y']

drop_cols = [
    'Unnamed: 0', 'barcode', 'Samples', 'Stage_x', 'Stage_y',
    'patient', 'sample', 'sample_submitter_id', 'sample_id'
]
X = merged_df.drop(columns=[col for col in drop_cols if col in merged_df.columns])

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))