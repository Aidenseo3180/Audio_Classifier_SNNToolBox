import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# Patch NumPy for SNNToolbox compatibility
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.long = int
np.unicode = str

import configparser
import tensorflow as tf
import shutil
################################################################################
# SNNToolbox Imports
from snntoolbox.bin.run import main
################################################################################
# Configure parameters for config
NUM_STEPS_PER_SAMPLE = 60  # Number of timesteps to run each sample (DURATION)
BATCH_SIZE = 32             # Affects memory usage. 32 -> 10 GB
NUM_TEST_SAMPLES = 1000      # Number of samples to evaluate or use for inference
CONVERT_MODEL = True      
################################################################################
# Paths
MODEL_FILENAME = 'audio_classifier.h5'
MODEL_NAME = MODEL_FILENAME.strip('.h5')
CURR_DIR = os.path.abspath('.')
ANN_MODEL_PATH = os.path.join(CURR_DIR, 'models', MODEL_FILENAME)
WORKING_DIR = os.path.join(CURR_DIR, 'snntb_audio_runs')
DATASET_DIR = os.path.join(CURR_DIR, 'data')
################################################################################
# Check paths exist and TODO: cleanup last run
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)  # Remove the entire directory
os.mkdir(WORKING_DIR)
assert os.path.exists(ANN_MODEL_PATH) ,'ERROR --> model path not found.'
assert os.path.exists(DATASET_DIR), 'ERROR --> data dir not found.'
assert len(os.listdir(DATASET_DIR)) > 1, 'ERROR --> data files in data dir not valid.'
################################################################################
# Copy model to working directory
os.system('cp {} {}'.format(ANN_MODEL_PATH, WORKING_DIR))  
################################################################################
# Generate Config file
config = configparser.ConfigParser()
config['paths'] = {
    'path_wd': WORKING_DIR,
    'dataset_path': DATASET_DIR,
    'filename_ann': MODEL_NAME,
    'runlabel': MODEL_NAME+'_'+str(NUM_STEPS_PER_SAMPLE)
}
config['tools'] = {
    'evaluate_ann': True,
    'parse': True,
    'normalize': True,
    'simulate': True
}
config['simulation'] = {
    'simulator': 'INI',  # INI Simulator - The SNN is automatically run during the conversion process
    'duration': NUM_STEPS_PER_SAMPLE,
    'num_to_test': NUM_TEST_SAMPLES,
    'batch_size': BATCH_SIZE,
    'keras_backend': 'tensorflow'
}
config['input'] = {
    'model_lib': 'keras',
    'dataset_format': 'npz',
    'normalize': True,
    'filename_test': 'x_test',
    'filename_test_labels': 'y_test',
    'filename_norm': 'x_norm',
    'dataflow_kwargs': "{'batch_size': 1}"
}
config['normalization'] = {
    'percentile': '98.2',
    'normalization_schedule': False,
    'weight_normalization': True 
}
config['output'] = {
    'verbose': 2,
    'plot_vars': {
    #    #'input_image',
    #    'spiketrains',
    #    'spikerates',
    #    'spikecounts',
        'operations',
        'normalization_activations',
        'activations',
        'correlation',
        'v_mem',
        'error_t'
    },
    'overwrite': True
}
# Write the configuration file
config_filepath = os.path.join(WORKING_DIR, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)
################################################################################
# Debug: Check the data files before conversion
print("Checking data files...")
norm_data = np.load(os.path.join(DATASET_DIR, 'x_norm.npz'))
test_data = np.load(os.path.join(DATASET_DIR, 'x_test.npz'))
test_labels = np.load(os.path.join(DATASET_DIR, 'y_test.npz'))

print(f"Norm data keys: {list(norm_data.keys())}")
print(f"Test data keys: {list(test_data.keys())}")
print(f"Test labels keys: {list(test_labels.keys())}")

print(f"Norm data shape: {norm_data['arr_0'].shape}")
print(f"Test data shape: {test_data['arr_0'].shape}")
print(f"Test labels shape: {test_labels['arr_0'].shape}")

# Load and check the model
model = tf.keras.models.load_model(ANN_MODEL_PATH)
print(f"Model input shape: {model.input_shape}")

# Print layer names to debug
print("\nModel layer names:")
for layer in model.layers:
    print(f"Layer name: {layer.name}, type: {type(layer).__name__}")

#Convert the model using SNNToolbox
if CONVERT_MODEL:
    main(config_filepath)
################################################################################
# Copy the resulting file to a snn_out dir.
print('Working Directory: '+WORKING_DIR)
print('Dataset Directory: '+DATASET_DIR)
print('SNN is located at: '+WORKING_DIR)
