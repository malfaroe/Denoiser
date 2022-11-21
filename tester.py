"""Unit for working out the mixing of two
sound files into one, step one of
the Data Engine Construktion project"""


import librosa
import numpy as np
import os
import soundfile as sf
import io
from scipy.io.wavfile import write
import random

import warnings
warnings.filterwarnings("ignore")


FILES_DIR = r"/Users/mauricioalfaro/Documents/mae_code/Denoiser/Data Engine/Sound"
NOISE_DIR = r"/Users/mauricioalfaro/Documents/mae_code/Denoiser/Data Engine/Noises"
GENERATED_DIR = r"/Users/mauricioalfaro/Documents/mae_code/Denoiser/Data Engine/MixedNoise"

FRAME_SIZE = 2048
HOP_SIZE = 512

def mixer_creator(files_dir, noise_dir, save_dir,
 n_fft = FRAME_SIZE, hop_length = HOP_SIZE):
    """Takea a file, extracts its specttrpgram and then reconstructs the file
    The idea is to make sure the reconstructions has the highest possible quality
    1. read the audio file
    2. Extract spectrogram
    3. Reconstruct
    4. save the reconstructed file
    5. Check quality"""
    
    #1. read audiofiles
    files_names= os.listdir(files_dir)
    noise_names = os.listdir(noise_dir)
    for file in files_names:
        #Loads the file
        numpy_file, sr_original = librosa.load(os.path.join(files_dir, file))
        print("Sounda loaded file length:", len(numpy_file))
        #Extract spectrogram
        spec_file = librosa.stft(numpy_file,  n_fft, hop_length)
        #Separate magnitude and phase
        file_magnitude, file_phase = librosa.magphase(spec_file)
        #Magnitude to dB
        file_mag_db = librosa.amplitude_to_db(file_magnitude, ref = np.max)

    for file in noise_names:
        #Loads the file
        noise_numpy_file, sr_original = librosa.load(os.path.join(noise_dir, file))
        print("Noise loaded file length:", len(noise_numpy_file))
        #recorto para ver si sirve
        noise_numpy_file = noise_numpy_file[:len(numpy_file)]
        print("New shape of noise:", noise_numpy_file.shape)

        #Extract spectrogram
        noise_spec_file = librosa.stft(noise_numpy_file,  n_fft, hop_length)
        #Separate magnitude and phase
        noise_file_magnitude, file_phase = librosa.magphase(noise_spec_file)
        #Magnitude to dB
        noise_file_mag_db = librosa.amplitude_to_db(noise_file_magnitude, ref = np.max)

        #REVERSE PROCESS
        # #Db to amplitude magnitude
        # reversed_magnitude = 100 * librosa.db_to_amplitude(file_mag_db, ref = 1.0) #multiplied by 100 to increase volume
        # audio_reverse = reversed_magnitude * file_phase
        # reconstructed_file = librosa.core.istft(audio_reverse, hop_length, n_fft)
        # #Save reconstructed file
        # save_path = os.path.join(save_dir, str(i) + ".wav")
        # sf.write(save_path, reconstructed_file, sr_original)
    print("Sound spec shape:", file_magnitude.shape)
    print("Noise spec shape:", noise_file_magnitude.shape)

    #Mixing
    mixed_magnitude = file_magnitude + 0 * noise_file_magnitude
    audio_reverse = mixed_magnitude * file_phase
    reconstructed_file = librosa.core.istft(audio_reverse, hop_length, n_fft)
    #Save reconstructed file
    save_path = os.path.join(save_dir, "tested_file.wav")
    sf.write(save_path, reconstructed_file, sr_original) 


    
# if __name__ == "__main__":
#     mixer_creator(FILES_DIR, NOISE_DIR, GENERATED_DIR, FRAME_SIZE, HOP_SIZE)
#     print("Done!") 

FRAME_SIZE = 2048
HOP_SIZE = 512
sound_file = r"/Users/mauricioalfaro/Documents/mae_code/Denoiser/Data Engine/Sound/84-121123-0001.flac"
noise_file = r"/Users/mauricioalfaro/Documents/mae_code/Denoiser/Data Engine/Noises/2-141682-A-36.wav"
loaded_sound, sr_sound = librosa.load(sound_file)
loaded_noise, sr_noise = librosa.load(noise_file)
print("Loaded sound file shape:", loaded_sound.shape)
print("Loaded noise file shape:", loaded_noise.shape)
print("Sample rates:", (sr_sound, sr_noise))
sound_spec_file = librosa.stft(loaded_sound,  n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
noise_spec_file = librosa.stft(loaded_noise,  n_fft = FRAME_SIZE, hop_length = HOP_SIZE)

print("sound  spec file shape:", sound_spec_file.shape)
print("noise spec file shape:", noise_spec_file.shape)

print(np.vstack(sound_spec_file).shape)


