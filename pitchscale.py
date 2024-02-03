import os

import numpy as np
import librosa
import soundfile as sf

# def add_white_noise(signal,noise_factor):
#     noise = np.random.normal(0, signal.std(), signal.size)
#     augmented_signal = signal + noise * noise_factor
#     return augmented_signal
#
# def time_strech(signal , rate):
#     return librosa.effects.time_stretch(signal,rate=rate)

def pitch_scale(signal,sr,num_semitones):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=num_semitones)

def process_folder(input_folder,output_folder,num_semitones):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder,filename)
        output_file_name = "Fm"+filename[2:]
        output_file_path = os.path.join(output_folder,output_file_name)

        audio , sr = librosa.load(input_file_path)

        pitched_audio = pitch_scale(audio , sr ,num_semitones)

        sf.write(output_file_path,pitched_audio,sr)


input_folder = "Test/minor/Em"
output_folder = "Test/minor/Fm"
num_semitones = 1

process_folder(input_folder,output_folder,num_semitones)

