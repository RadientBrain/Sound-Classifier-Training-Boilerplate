# sound_detection.py

import os

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def create_spectrogram(audio_file):
    y, sr = librosa.load(audio_file)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect

def predict_sound(model, audio_file):
    spect = create_spectrogram(audio_file)
    spect = np.expand_dims(spect, axis=0)
    prediction = model.predict(spect)
    return np.argmax(prediction)

def main():
    model_path = 'models/model.h5'
    audio_file = 'data/test/test.wav'

    model = load_model(model_path)
    prediction = predict_sound(model, audio_file)

    if prediction == 0:
        print('Sound of glass breaking detected.')
    elif prediction == 1:
        print('Sound of a scream detected.')
    else:
        print('No sound detected.')

if __name__ == '__main__':
    main()