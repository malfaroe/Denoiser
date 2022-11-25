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


FILES_DIR = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\Sound"
NOISE_DIR = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\Noises"
GENERATED_DIR = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\MixedNoise"

FRAME_SIZE = 2048
HOP_SIZE = 512

def equalize_arrays(x, y):
    """Checks if two arrays have different number of elements y axis 1 
    and equalize them"""
    x = np.reshape(x, (1, len(x)))
    y = np.reshape(y, (1, len(y)))
    if x.shape[1] < y.shape[1]:
        new_x = np.zeros((x.shape[0], y.shape[1]))
        for i in range(x.shape[0]):
           new_x[i] = np.append(x[i], x[i][:(y.shape[1] - x.shape[1])], axis = 0)
        x = new_x
        print("New shape x:", x.shape)
    elif x.shape[1] > y.shape[1]:
        new_y = np.zeros((y.shape[0], x.shape[1]))
        for i in range(y.shape[0]):
           new_y[i] = np.append(y[i], y[i][:(x.shape[1] - y.shape[1])], axis = 0)
        y = new_y
        print("New shape y:", y.shape)

    return x.reshape(len(x[0])), y.reshape(len(y[0]))



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
        file_s = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\Sound\84-121123-0001.flac"
        # sound_file, sr_original = librosa.load(os.path.join(files_dir, file))
        sound_file, sr_original = librosa.load(file_s)
        #Loads a random file from the noise directory
        # random_noise_file = np.random.choice(noise_names)
        random_noise_file = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\Noises\2-152895-A-31.wav"
        noise_file, sr_original = librosa.load(random_noise_file)
        print("Shape of sound file:", np.array(sound_file).shape)
        print("Shape of noise file:", noise_file.shape)

        #Equalize the files
        sound_file, noise_file = equalize_arrays( sound_file, noise_file)

        #Extract spectrogram sound file
        spec_file = librosa.stft(sound_file,  n_fft, hop_length)
        #Separate magnitude and phase
        sound_file_magnitude, sound_file_phase = librosa.magphase(spec_file)
        #Magnitude to dB
        file_mag_db = librosa.amplitude_to_db(sound_file_magnitude, ref = np.max)

    
        #Extract spectrogram noise file
        noise_spec_file = librosa.stft(noise_file,  n_fft, hop_length)
        #Separate magnitude and phase
        noise_file_magnitude, noise_file_phase = librosa.magphase(noise_spec_file)
        #Magnitude to dB
        noise_file_mag_db = librosa.amplitude_to_db(noise_file_magnitude, ref = np.max)

        #Mix the sound magnitude and noise
        mixed_magnitude_file = sound_file_magnitude + 0.5 * noise_file_magnitude

        #Reconstruct
        audio_reverse =    mixed_magnitude_file * sound_file_phase
        reconstructed_file = librosa.core.istft(audio_reverse, hop_length, n_fft)
        # print("Dtype of reconstructed file:", np.dtype(reconstructed_file))
        # #Save reconstructed file
        save_path = os.path.join(save_dir, "tester_mae.wav")
        sf.write(save_path, reconstructed_file, sr_original)
  



    
# if __name__ == "__main__":
#     mixer_creator(FILES_DIR, NOISE_DIR, GENERATED_DIR, FRAME_SIZE, HOP_SIZE)
#     print("Done!") 

FRAME_SIZE = 2048
HOP_SIZE = 512
def remove_ds_store(lst):
        """remove mac specific file if present"""
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')
            print("one file removed")

        return lst
files_names= os.listdir(FILES_DIR)
noise_names = os.listdir(NOISE_DIR)
print(files_names)
print(noise_names)
remove_ds_store(files_names)
remove_ds_store(noise_names)
mixer_creator(files_dir = FILES_DIR, noise_dir = NOISE_DIR, save_dir = GENERATED_DIR,
n_fft = FRAME_SIZE, hop_length = HOP_SIZE)

###NEXT
"""1. Code refactoring
2. Multiple files generation
3. Integrate to Data Engine"""