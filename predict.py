import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = tf.keras.models.load_model('sound_event_model.h5')

# Load the label encoder used during training
# You should save and load the label encoder used during training for consistent results
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')  # Assuming you saved class names

def extract_features(file_path, window_size=2048, hop_size=512, sr=None):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_size, n_fft=window_size)
        return mfccs.T  # Transpose so that time is along rows
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_sound_events(file_path, model, label_encoder):
    feature = extract_features(file_path)
    
    if feature is None:
        return "Error extracting features"
    
    # Ensure the feature shape matches (1, 862, 13)
    feature = feature.reshape(1, -1, feature.shape[-1])  # (1, 862, 13)
    
    # Predict using the model
    predictions = model.predict(feature.astype(np.float32))
    
    # Post-process predictions to get events
    events = post_process_predictions(predictions, feature.shape[1], 512, label_encoder)
    
    return events

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
file_path = './data/soundscapes/1333.wav'
predicted_events = predict_sound_events(file_path, model, label_encoder)
print(predicted_events)
