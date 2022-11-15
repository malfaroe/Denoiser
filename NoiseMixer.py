import librosa
import numpy as np
import os
import soundfile as sf
import io
from scipy.io.wavfile import write
import random


"""
MIXER OF NOISE FILES

1. Reads x random audio noise files from original folder
2. Converts them to spec
3. Mixing
4. Back to audio to final folder 
"""

files_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\Noises"
generated_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\Data Engine\MixedNoise"
# random_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\randomNoise"

FRAME_SIZE = 2048
HOP_SIZE = 512
#1. Lee archivos
files_names= os.listdir(files_dir)

####RANDOM NOISE INJECTION UNIT ---SUSPENDED
# #3. Load a to a numpy file each file
# spec_files = [] #lista para guardar todos losa arrays en una matriz
# for i, file in enumerate(files_names):
#     numpy_file, sr_original = librosa.load(os.path.join(files_dir, file))
#     spec_file = librosa.stft(numpy_file,  n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
#     print("File read size (nr_freq_bins, nr_time_bins):", spec_file.shape )
#     #generate the random noise
#     noise_spec = np.random.randn(spec_file.shape[0], spec_file.shape[1])
#     #Back to wav
#     S = np.abs(spec_file)**2
#     y_inv = librosa.griffinlim(S)
#     noise = librosa.griffinlim(noise_spec)
#     #Mixing the file
#     noised_file = y_inv + noise
#     #Save
#     save_path = os.path.join(random_dir, str(i) + ".wav")
#     sf.write(save_path, noised_file, sr_original)
    
###########################################

def reconstructor(files_dir, save_path,
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
    for i, file in enumerate(files_names):
        #Loads the file
        numpy_file, sr_original = librosa.load(os.path.join(files_dir, file))
        #Extract spectrogram
        spec_file = librosa.stft(numpy_file,  n_fft, hop_length)
        #Separate magnitude and phase
        file_magnitude, file_phase = librosa.magphase(spec_file)
        #Magnitude to dB
        file_mag_db = librosa.amplitude_to_db(file_magnitude, ref = np.max)

        #REVERSE PROCESS
        #Db to amplitude magnitude
        reversed_magnitude = 100 * librosa.db_to_amplitude(file_mag_db, ref = 1.0) #multiplied by 100 to increase volume
        audio_reverse = reversed_magnitude * file_phase
        reconstructed_file = librosa.core.istft(audio_reverse, hop_length, n_fft)
        #Save reconstructed file
        save_path = os.path.join(random_dir, str(i) + ".wav")
        sf.write(save_path, reconstructed_file, sr_original)


def noiser(files_dir, save_path,  
n_fft = FRAME_SIZE, hop_length = HOP_SIZE):
    """Reads a clean audio file and injects noise on it,then saves it"""
    #1. read audiofiles
    files_names= os.listdir(files_dir)
    for i, file in enumerate(files_names):
        #Loads the file
        numpy_file, sr_original = librosa.load(os.path.join(files_dir, file))
        #Extract spectrogram
        spec_file = librosa.stft(numpy_file,  n_fft, hop_length)
        #Separate magnitude and phase
        file_magnitude, file_phase = librosa.magphase(spec_file)
        #Magnitude to dB
        file_mag_db = librosa.amplitude_to_db(file_magnitude, ref = np.max)
        #2. generates the random noise
        noise_spec = np.random.randn(spec_file.shape[0], spec_file.shape[1])
        #Mixing...
        mixed_magnitude = file_magnitude + noise_spec
        #Reversing...
        audio_reverse = mixed_magnitude * file_phase
        reconstructed_file = librosa.core.istft(audio_reverse, hop_length, n_fft)
        #Save
        save_path = os.path.join(random_dir, str(i) + ".wav")
        sf.write(save_path, reconstructed_file, sr_original)

    
def spec_tensor(files_dir, n_fft = FRAME_SIZE, hop_length = HOP_SIZE):
    """Take every audio file from a directory, extracts its
    spectrogram and saves it to a container/tensor"""
    files_names= os.listdir(files_dir)
    print("Files:", files_names)
    audio_spec_tensor = []
    for i, file in enumerate(files_names):
        #Loads the file
        numpy_file, sr_original = librosa.load(os.path.join(files_dir, file))
        #Extract spectrogram
        spec_file = librosa.stft(numpy_file,  n_fft, hop_length)
        print(spec_file.shape)
        audio_spec_tensor.append(spec_file)
    #final_tensor = np.vstack(audio_spec_tensor)
    #print("Tensor created. Final shape:", final_tensor.shape)




def noise_mixer(files_dir,  
n_fft = FRAME_SIZE, hop_length = HOP_SIZE):
    files_names= os.listdir(files_dir)
    #Create N individual mized files
    i = 0
    N = 10
    n = 5   #we will pick 5 random files each time
    for i in range(N):
        selected_for_mix = random.sample(files_names, n)
        print(selected_for_mix)
        #Loads each one file

    


    




if __name__ == "__main__":
    # reconstructor(files_dir, random_dir)
    # noiser(files_dir, random_dir)
    #noise_mixer(files_dir)
    spec_tensor(files_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\data_toy")




print("Done!")