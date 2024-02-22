###### IMPORTS ################
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#### LOADING THE VOICE DATA FOR VISUALIZATION ###
walley_sample = r"D:\AI Assistant\Wake word detection\WakeWordDetection\audio_data\voice_10.wav"
data, sample_rate = librosa.load(walley_sample)

# Plotting the waveform
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

##### VISUALIZING MFCC #######
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print("Shape of mfcc:", mfccs.shape)

plt.title("MFCC")
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.show()

# Dictionary to map class labels
class_labels = {
    "voice": 1,
    "background": 0
}
all_data = []

# Construct directory path
dir_path_voice = os.path.join(os.path.dirname(__file__), "audio_data")
dir_path_background = os.path.join(
    os.path.dirname(__file__), "background_sound")

# Iterate over each directory and load audio files
for class_label, dir_path in [("voice", dir_path_voice), ("background", dir_path_background)]:
    # Check if the directory exists
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} not found.")
        continue

    # Iterate over each file in the directory
    for file_name in os.listdir(dir_path):
        # Check if the file is a WAV file
        if file_name.endswith(".wav"):
            # Construct full file path
            file_path = os.path.join(dir_path, file_name)

            try:
                # Load audio file
                audio, sample_rate = librosa.load(file_path)

                # Extract features (MFCC)
                mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                mfcc_processed = np.mean(mfcc.T, axis=0)  # Calculate mean MFCC

                # Append features and class label to the list
                all_data.append([mfcc_processed, class_labels[class_label]])

                print(f"Processed {file_name} in class {class_label}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Convert the list to a DataFrame
df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# Save the DataFrame to a CSV file
df.to_csv("final_audio_data.csv", index=False)
