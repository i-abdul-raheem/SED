import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Function to extract MFCC features using sliding windows
def extract_features(file_path, window_size=2048, hop_size=512, sr=None):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_size, n_fft=window_size)
        return mfccs.T  # Transpose so that time is along rows
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load labels from a .txt file
def load_labels(file_path):
    labels = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 3:
                    labels.append(parts[2])
                else:
                    print(f"Unexpected label format in {file_path}: {line}")
    except Exception as e:
        print(f"Error loading labels from {file_path}: {e}")
    return labels

# Prepare dataset from audio and label files
def prepare_dataset(data_path, labels_path, features_file, labels_file):
    if os.path.exists(features_file) and os.path.exists(labels_file):
        print("[+] Saved files found")
        features = np.load(features_file)
        labels = np.load(labels_file, allow_pickle=True)
    else:
        features = []
        labels = []
        files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
        
        for file_name in tqdm(files, desc="Processing audio files"):
            audio_path = os.path.join(data_path, file_name)
            label_path = os.path.join(labels_path, file_name.replace('.wav', '.txt'))
            if os.path.exists(label_path):
                feature = extract_features(audio_path)
                if feature is not None:
                    labels_list = load_labels(label_path)
                    if labels_list:
                        features.append(feature)
                        labels.append(labels_list[0])
                    else:
                        print(f"No labels found in {label_path}. Skipping this file.")
                else:
                    print(f"Feature extraction failed for {audio_path}. Skipping this file.")
            else:
                print(f"Label file {label_path} does not exist. Skipping this file.")
        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels)
        np.save(features_file, features)
        np.save(labels_file, labels)
    
    return features, labels

# Load training and validation data
print("\n----- Load training and validation data -----\n")
X_train, y_train = prepare_dataset('./data/soundscapes', './data/soundscapes', './npy/X_train.npy', './npy/y_train.npy')
X_val, y_val = prepare_dataset('./data/soundscapes_val', './data/soundscapes_val', './npy/X_val.npy', './npy/y_val.npy')

print(f'X_train dtype: {X_train.dtype}, shape: {X_train.shape}')
print(f'y_train dtype: {y_train.dtype}, shape: {y_train.shape}')
print(f'X_val dtype: {X_val.dtype}, shape: {X_val.shape}')
print(f'y_val dtype: {y_val.dtype}, shape: {y_val.shape}')

# Debug: Inspect original labels
print("\nUnique labels in y_train:", np.unique(y_train))
print("Unique labels in y_val:", np.unique(y_val))

# Fit the Label Encoder on combined labels
print("\n----- Fit the Label Encoder on combined labels -----\n")
all_labels = np.concatenate((y_train, y_val))
print("Unique labels in combined dataset:", np.unique(all_labels))
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
np.save('./npy/classes.npy', label_encoder.classes_)
print("Classes learned by LabelEncoder:", label_encoder.classes_)

# Encode labels
print("\n----- Encode labels -----\n")
y_train_int = label_encoder.transform(y_train)
y_val_int = label_encoder.transform(y_val)

print("First 10 integer-encoded labels in y_train:", y_train_int[:10])
print("First 10 integer-encoded labels in y_val:", y_val_int[:10])

# Convert labels to one-hot encoding
print("\n----- Convert labels to one-hot encoding -----\n")
num_classes = len(label_encoder.classes_)
y_train_encoded = to_categorical(y_train_int, num_classes=num_classes)
y_val_encoded = to_categorical(y_val_int, num_classes=num_classes)

# Ensure labels are correctly formatted
print(f'y_train_encoded dtype: {y_train_encoded.dtype}, shape: {y_train_encoded.shape}')
print(f'y_val_encoded dtype: {y_val_encoded.dtype}, shape: {y_val_encoded.shape}')

# Check for all-zero rows
print("Any all-zero rows in y_train_encoded:", np.any(np.all(y_train_encoded == 0, axis=1)))
print("Any all-zero rows in y_val_encoded:", np.any(np.all(y_val_encoded == 0, axis=1)))

# Split data into training and testing sets
print("\n----- Split data into training and testing sets -----\n")
X_train_final, X_test, y_train_final, y_test = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42
)

print("X_train_final shape:", X_train_final.shape)
print("X_test shape:", X_test.shape)
print("y_train_final shape:", y_train_final.shape)
print("y_test shape:", y_test.shape)

# Define the model
print("\n----- Define the model -----\n")
model = Sequential([
    Input(shape=(X_train_final.shape[1], X_train_final.shape[2])),  # (862, 13)
    LSTM(128, return_sequences=False),  # LSTM layer to process sequence data
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
print("\n----- Compile the model -----\n")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with a progress bar
print("\n----- Train the model with a progress bar -----\n")
history = model.fit(
    X_train_final, y_train_final,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val_encoded),
    verbose=1
)

# Evaluate the model
print("\n----- Evaluate the model -----\n")
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the model
print("\n----- Save the model -----\n")
model.save('sound_event_model.h5')

# Function to predict multiple sound events in a file
def predict_sound_events(file_path, model, label_encoder):
    # Extract MFCC features
    feature = extract_features(file_path)
    
    if feature is None:
        return "Error extracting features"
    
    # Ensure the feature shape matches (1, 862, 13)
    feature = feature.reshape(1, -1, feature.shape[-1])  # (1, 862, 13)
    
    # Predict using the model
    predictions = model.predict(feature.astype(np.float32))
    
    # Post-process predictions to get events
    events = post_process_predictions(predictions, feature.shape[0], 512, label_encoder)
    
    return events

# Post-process function to extract events
def post_process_predictions(predictions, num_frames, hop_length, label_encoder):
    events = []
    current_event = None
    current_label = None
    
    for i, prediction in enumerate(predictions[0]):  # [0] to get the single batch
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        if current_label is None:
            current_label = predicted_label
            onset = i * hop_length / 22050  # Assume a sample rate of 22050 Hz
            
        if predicted_label != current_label:
            offset = i * hop_length / 22050
            events.append({"onset": onset, "offset": offset, "label": current_label})
            current_label = predicted_label
            onset = i * hop_length / 22050
            
    if current_label is not None:
        offset = num_frames * hop_length / 22050
        events.append({"onset": onset, "offset": offset, "label": current_label})
    
    return events

# Example usage:
print("\n----- Example prediction -----\n")
file_path = './data/soundscapes/9999.wav'
predicted_events = predict_sound_events(file_path, model, label_encoder)
print(predicted_events)
