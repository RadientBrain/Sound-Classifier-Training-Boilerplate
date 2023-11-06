# data_preprocessing.py

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

def create_spectrogram(audio_file):
    y, sr = librosa.load(audio_file)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect

def process_data(data_dir, save_dir):
    labels = []
    features = []

    for folder in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, folder)):
            if file.endswith('.wav'):
                spect = create_spectrogram(os.path.join(data_dir, folder, file))
                features.append(spect)
                labels.append(folder)

    features = np.array(features)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)