"""Data creation and optimzation for denoiser project"""

import librosa
import numpy as np
import os
import soundfile as sf
import io
from scipy.io.wavfile import write
import random


FILES_DIR = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\Sound"
NOISE_DIR = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\Noises"
GENERATED_DIR = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\MixedNoise"


FRAME_SIZE = 2048
HOP_SIZE = 512


class DataEngine():

    def __init__(self, files_dir, noise_dir, generated_dir):
        self.files_dir = files_dir
        self.noise_dir = noise_dir
        self.generated_dir = generated_dir


    def audio_numpy_matrix(self, files_directory):
        """Generates a matrix containing all the sound files
        in the form of frames stackec via vstack"""
        files_names= os.listdir(files_directory)
        numpy_matrix = []
        print("Starting...")
        for i, file in enumerate(files_names):
            #Loads the file
            loaded_file, sr_original = librosa.load(os.path.join(files_directory, file))
            numpy_matrix.append(self.frames_generate(loaded_file, FRAME_SIZE, HOP_SIZE))
        
        numpy_matrix = np.vstack(numpy_matrix)
        print(numpy_matrix.shape)
        print("Numpy matrix created")
        return numpy_matrix


    def frames_generate(self, loaded_file, frame_size, hop_size):
        """Take a file generated via librosa.load
        and create nb frames"""
        file_length = loaded_file.shape[0]
        frames = [loaded_file[start:start + frame_size]
        for start in range(0, file_length - frame_size +1, hop_size)]
        frames_array = np.vstack(frames)
        return frames_array


    def noise_blender(self, files_dir, noise_dir, generated_dir, frame_size, hop_size):
        """Generates a big numpy file  randomly mixing two folders of sound 
        and noise folders"""
        #Creates the loaded files for sound and noise folders
        sound_numpy_matrix = self.audio_numpy_matrix(files_dir)
        print("Sound numpy matrix done")
        noise_numpy_matrix = self.audio_numpy_matrix(noise_dir)
        print("Noise numpy matrix done")
        mixed_sound = np.zeros((sound_numpy_matrix.shape[0], sound_numpy_matrix.shape[1]))
        #Randomly mixing the files...
        for i in range(sound_numpy_matrix.shape[0]):
            j = np.random.choice(len(noise_numpy_matrix[0]))
            mixed_sound[i, :] = sound_numpy_matrix[i, :] + noise_numpy_matrix[j, :]
        
        print("Shape of the mixed sound matrix:", mixed_sound.shape)



        print("Test passed")


if __name__ == "__main__":
    test_file = np.random.randn(8000)
    engine = DataEngine(files_dir = FILES_DIR, noise_dir = NOISE_DIR,
    generated_dir = GENERATED_DIR)
    # frames_test = engine.frames_generate(test_file, FRAME_SIZE, HOP_SIZE)
    # print(frames_test.shape)
    #engine.audio_numpy_matrix()
    engine.noise_blender(files_dir = FILES_DIR, noise_dir = NOISE_DIR,
    generated_dir = GENERATED_DIR, frame_size = FRAME_SIZE, hop_size = HOP_SIZE)
