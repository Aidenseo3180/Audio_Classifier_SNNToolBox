"""
Audio classification model using TensorFlow/Keras for directory-based audio datasets.
"""
################################################################################
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.utils import to_categorical
import librosa  # We'll use librosa for audio processing
from sklearn.model_selection import train_test_split
################################################################################
# Clean up settings
REMOVE_MODEL = True
REMOVE_DATA = True
TRAIN_NEW_MODEL = True
USE_GPUS = False
N_GPUS = 4
################################################################################
# GLOBAL VARS
DATASET_DIR = ''  # TODO: Update this with your audio dataset path
OUTPUT_CLASSES = 10  # Update based on number of folders/classes
BATCH_SIZE = 32 #32
EPOCHS = 50
LEARNING_RATE = 0.001

# Audio processing parameters
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_MFCC = 40  # number of MFCC features

# Split ratios
TRAIN_PERC = 0.7
VAL_PERC = 0.15
TEST_PERC = 0.15
################################################################################
# Paths
MODEL_OUT_DIR = os.path.join(os.path.abspath('..'), 'models')
WORKING_DIR = os.path.join(os.path.abspath('..'),'data')
# Output filenames
MODEL_NAME = 'test_classifier.h5'
# Print the dirs
print('LOG --> MODEL_OUT_DIR: '+str(MODEL_OUT_DIR))
print('LOG --> DATASET_DIR: '+str(DATASET_DIR))
# Check that dirs exist if not create
if not os.path.exists(MODEL_OUT_DIR):
    os.mkdir(MODEL_OUT_DIR)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
# Cleanup last run
if os.path.exists(os.path.join(MODEL_OUT_DIR, MODEL_NAME)) and REMOVE_MODEL:
    os.remove(os.path.join(MODEL_OUT_DIR, MODEL_NAME))
if os.path.exists(os.path.join(WORKING_DIR, 'x_test.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'x_test.npz'))
if os.path.exists(os.path.join(WORKING_DIR, 'y_test.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'y_test.npz'))
if os.path.exists(os.path.join(WORKING_DIR, 'x_norm.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'x_norm.npz'))
################################################################################
# Distribute on multiple GPUS
if USE_GPUS:
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(device_type)
    devices_names = [d.name.split('e:')[1] for d in devices]
    print(devices_names)
    strategy = tf.distribute.MirroredStrategy(devices=devices_names[:N_GPUS])
else:
    device_type = 'CPU'
    devices = tf.config.experimental.list_physical_devices(device_type)
    device_names = [d.name.split('e:')[1] for d in devices]
    strategy = tf.distribute.OneDeviceStrategy(device_names[0])
################################################################################
# Helper Functions
def extract_features(file_path, duration=DURATION, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Extract MFCC features from audio file."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        # Pad or truncate audio
        if len(audio) < duration * sr:
            audio = np.pad(audio, (0, duration * sr - len(audio)))
        else:
            audio = audio[:duration * sr]
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)
        
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_audio_data(dataset_dir):
    """Load audio data from directory structure where folder names are labels."""
    features = []
    labels = []
    label_names = []
    
    # Get list of folders (labels)
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    folders.sort()  # Ensure consistent ordering
    
    print(f"Found {len(folders)} classes: {folders}")
    
    for i, folder in enumerate(folders):
        label_names.append(folder)
        folder_path = os.path.join(dataset_dir, folder)
        
        # Get all audio files in the folder
        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

        # audio_files = audio_files # read 100 elements from each
        
        print(f"Processing {len(audio_files)} files in folder {folder}")
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            mfcc_features = extract_features(file_path)
            
            if mfcc_features is not None:
                features.append(mfcc_features)
                labels.append(i)
    
    return np.array(features), np.array(labels), label_names

def save_data_as_npz(features, labels, suffix, split=None):
    """Saves data as separate feature and label files."""
    global WORKING_DIR
    if split:
        # For norm data, ensure the correct key name
        np.savez_compressed(os.path.join(WORKING_DIR, 'x_{}'.format(suffix)),
                            arr_0=features[:split])  # Use 'arr_0' as the key
        print(f"Saved x_{suffix}.npz with shape {features[:split].shape}")
    else:
        # For test data, save full features and labels separately
        np.savez_compressed(os.path.join(WORKING_DIR, 'x_{}'.format(suffix)),
                            arr_0=features)  # Use 'arr_0' as the key
        np.savez_compressed(os.path.join(WORKING_DIR, 'y_{}'.format(suffix)),
                            arr_0=labels)    # Use 'arr_0' as the key
        print(f"Saved x_{suffix}.npz with shape {features.shape}")
        print(f"Saved y_{suffix}.npz with shape {labels.shape}")
    return features, labels
################################################################################
# Dataset Import
print("Loading audio data from folders...")
X, y, label_names = load_audio_data(DATASET_DIR)
print(f"Total samples loaded: {len(X)}")
print(f"Feature shape: {X[0].shape}")
print(f"Classes: {label_names}")

# Convert labels to one-hot encoding
y_categorical = to_categorical(y, num_classes=OUTPUT_CLASSES)

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_categorical, test_size=TEST_PERC, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_PERC/(1-TEST_PERC), random_state=42, stratify=y_train_val.argmax(axis=1)
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Save test set
save_data_as_npz(X_test, y_test, 'test', None)

# Save normalization data (subset of training data)
# We use 100 samples for normalization (adjust as needed)
save_data_as_npz(X_train, None, 'norm', 100)

input_shape = X_train.shape[1:]  # Should be (n_mfcc,)
################################################################################
# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_OUT_DIR, MODEL_NAME),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=0,
    mode='min'
)
callbacks = [checkpoint, early_stop]
################################################################################
# Model Construction
print('LOG --> Input Shape: '+str(input_shape))

hidden_size1 = 128
hidden_size2 = 64

with strategy.scope():
    # Input Layer
    inputs = tf.keras.Input(input_shape)
    
    # Flatten the input to match the PyTorch model structure
    layer = tf.keras.layers.Flatten(name='flatten')(inputs)
    
    # Fully connected layers matching the PyTorch architecture
    layer = tf.keras.layers.Dense(units=hidden_size1, activation='relu', name='dense1')(layer)
    layer = tf.keras.layers.Dense(units=hidden_size2, activation='relu', name='dense2')(layer)
    outputs = tf.keras.layers.Dense(units=OUTPUT_CLASSES, activation='softmax', name='dense3')(layer)
    
    # Instantiate the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        run_eagerly=True
    )

# Show the model summary
model.summary()

# Print detailed information about shapes
print(f"Model input shape: {model.input_shape}")
print(f"Training data shape: {X_train.shape}")
print(f"Normalization data shape: {X_train[:5].shape}")

if TRAIN_NEW_MODEL:
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, 
        verbose=2,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        batch_size=BATCH_SIZE
    )

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(MODEL_OUT_DIR, 'training_loss_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to: {plot_path}")
    
    # Also create an accuracy plot
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the accuracy plot
    accuracy_plot_path = os.path.join(MODEL_OUT_DIR, 'training_accuracy_plot.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    print(f"Training accuracy plot saved to: {accuracy_plot_path}")
    
    # Close the figures to free memory
    plt.close('all')

else:
    model.load_weights(os.path.join(MODEL_OUT_DIR, MODEL_NAME))

# Evaluate the model on the test data
inference_start = time.time()
loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=2
)
inference_end = time.time()
total_inference_time = inference_end - inference_start
print('INFERENCE PERFORMED ON {} AUDIO SAMPLES IN BATCHES OF {}'.format(len(X_test), BATCH_SIZE))
print('EVALUATION LATENCY: {}'.format(total_inference_time))
print('EVALUATION LOSS: {}, EVALUATION ACC: {}'.format(loss,acc))
