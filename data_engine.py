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
        for file in files_names:
            #Loads the file
            loaded_file, sr_original = librosa.load(os.path.join(files_directory, file))
            numpy_matrix.append(self.frames_generate(loaded_file, FRAME_SIZE, HOP_SIZE))
        
        numpy_matrix = np.vstack(numpy_matrix)
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
        print("Shape sound numpy:", sound_numpy_matrix.shape)
        noise_numpy_matrix = self.audio_numpy_matrix(noise_dir)
        print("Shape noise numpy:", noise_numpy_matrix.shape)
        print("Noise numpy sample:", noise_numpy_matrix[100:103])

        mixed_sound = np.zeros((sound_numpy_matrix.shape[0], sound_numpy_matrix.shape[1]))
        #Randomly mixing the files...
        for i in range(sound_numpy_matrix.shape[0]):
            # j = np.random.choice(len(noise_numpy_matrix[0]))
            j = np.random.choice(noise_numpy_matrix.shape[0])
            mixed_sound[i, :] = sound_numpy_matrix[i, :] + 0.4 * noise_numpy_matrix[j, :]

        print("Total files processsed:", i)
        return mixed_sound

        """The problem is in the queality of the wav converter unit"""

    def blended_wav_saver(self, numpy_file, generated_dir):
        """Converts a numpy file into a wav file.
        Args:
        numpy_file: numpy matrix containing frames from sound files
        generated_dir: folder for saving the converted file
        Method takes de file, reshapes it as one single long sequence
        and converts into a wav sound file"""
        numpy_file_reshaped = numpy_file.reshape(1, numpy_file.shape[0] * numpy_file.shape[1])
        print("Shape of the unified file:", numpy_file_reshaped.shape )
        # librosa.output.write_wav(generated_dir + "noisy_long_sound_file.wav", numpy_file[0,:])
        save_path = os.path.join(generated_dir, "noisy_sound.wav")
        # sf.write(save_path, numpy_file_reshaped[0,:], 22050, 'PCM_24')
        sf.write(save_path, numpy_file_reshaped[0,:], 22050)
        print("Blended  wav file created")


   


    


if __name__ == "__main__":
    engine = DataEngine(files_dir = FILES_DIR, noise_dir = NOISE_DIR,
    generated_dir = GENERATED_DIR)
    numpy_mixed_file = engine.noise_blender(files_dir = FILES_DIR, noise_dir = NOISE_DIR,
    generated_dir = GENERATED_DIR, frame_size = FRAME_SIZE, hop_size = HOP_SIZE)
    engine.blended_wav_saver(numpy_mixed_file, generated_dir= GENERATED_DIR)
    
    print("Done...")


