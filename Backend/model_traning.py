
# Real-Time Noisy Environment Sound Classifier - Training Script


import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Paths


DATASET_PATH = "Backend/model/dataset/UrbanSound8k"  # Path to the audio folder
METADATA_PATH = "Backend/model/dataset/UrbanSound8K.csv"  # Path to metadata CSV


#Load metadata


metadata = pd.read_csv(METADATA_PATH)
print("Total samples in dataset:", len(metadata))


# Feature Extraction (MFCC)


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=2.5, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print("Error loading", file_path, e)
        return None

features = []

print("Extracting features...")
for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
    file_name = os.path.join(DATASET_PATH, f"fold{row.fold}", row.slice_file_name)
    class_label = row["class"]
    data = extract_features(file_name)
    if data is not None:
        features.append([data, class_label])

# Convert to DataFrame
features_df = pd.DataFrame(features, columns=["feature", "label"])
print(features_df.head())


# Prepare Training Data


X = np.array(features_df['feature'].tolist())
y = np.array(features_df['label'].tolist())

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Build Neural Network

model = Sequential([
    Dense(256, input_shape=(40,), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the Model


history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save the Model


model.save("urbansound8k_model.h5")
print("Model saved as urbansound8k_model.h5")

# Also save label encoder to map predictions back to class names
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Label encoder saved as label_encoder.pkl")
