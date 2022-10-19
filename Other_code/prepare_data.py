
import os
import librosa
import args

from data_tools import audio_files_to_numpy


def create_data(voice_dir, noise_dir, spectrogram_dir, time_serie_dir, sound_dir,
sample_rate, min_duration, frame_length, hop_length_frame,hop_length_frame_noise,
nb_samples, n_ftt,  hop_length_nftt):
    """ Creates the data for training: 
Data for training: random blend of clean and noisy files
input: clean and noisy sound files
output: spectrograms of blended, noisy and clean files,
complex phase and time_series files  """
    #1. Listing and cleaning the directories

    list_noise_files = os.listdir(noise_dir)
    list_voice_files = os.listdir(voice_dir)

    def remove_ds_store(lst):
        if ".DS_Store" in lst:
            lst.remove(".DS_Store")
        return lst

    #Remove de ds store file if in the directory
    
    list_noise_files = remove_ds_store(noise_dir)
    list_voice_files = remove_ds_store(voice_dir)

    #2. Convert noise and clean files to numpy using librosa´s load method
    noise = audio_files_to_numpy(noise_dir, 
    list_noise_files , frame_length, hop_length_frame_noise, min_duration)

    #3. Randomly mix voice and noise to generate  noisy files

    #4. Save the audio generated as wav

    #
list_voice_files = os.listdir(args.CLEAN_VOICE_DIR)
numpy_data = audio_files_to_numpy(args.CLEAN_VOICE_DIR, list_voice_files)
print("Array shape:", len(numpy_data))

