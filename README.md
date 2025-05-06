# Audio_Classifier_SNN
Using SNNToolBox for ANN - SNN conversion for audio classification.

# How to Use
## Step 1
Install Anaconda, and create virtual environments using ann.yml modelgen folder, and snntb.yml from snntb folder.   
These libraries have been updated to the latest working versions.  

## Step 2
Use lenet_audio.py in modelgen, fill out the path of the dataset.  
If you want to use it for a different purpose, you can simply give a new path of the dataset.  
You can change the structure of the ANN in lenet_audio.py file.  
Then, activate conda environment of ANN, and run this file using 'python lenet_audio.py'.  

## Step 3
After the run, inside modelgen folder, it will generate x_norm.npz, x_test.npz, and y_test.npz files inside the data folder, and audio_classifier.h5 inside the models folder.  
You'll need to manually move these files over to the data and models folder from snntb folder.  
Then, activate conda environment of snntb, and run audio_snntb.py to generate SNN equivalent.  
Currently, it is designed to generate INI simulator. If you want to create one for other simulators such as Loihi, make sure to change the INI to Loihi in audio_snntb.py.  

## Step 4
After running snntb.py, it will generate an output depending on the simulator type.  
For INI, it will create figures in snntb_audio_runs folder, and .h5 file of SNN model.  
For other types of simulator, it may generate different outputs.  
