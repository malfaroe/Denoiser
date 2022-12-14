import librosa
import numpy as np
import os
import soundfile as sf
import io
from scipy.io.wavfile import write

print("Completed")

"""
Preparatory experiemntal module#1:
Puts noise into a file and saves the generated file: 
1. Reads the clean file from the files_dir directory
2. Extracts the spectrogram from thefile
3. Generates a random file the same size of the clean file spec
4. Changes generated files back to wav
5. Mixes the files
5. Saves the "noised" file in the random noise dir"""

files_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\data_toy"
generated_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\generated"
random_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\randomNoise"

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
    The idea is to make sure the reconstructions has the hights possible quality
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

    




if __name__ == "__main__":
    # reconstructor(files_dir, random_dir)
    noiser(files_dir, random_dir)




print("Done!")