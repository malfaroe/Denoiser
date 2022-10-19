import librosa
import numpy as np
import os
import soundfile as sf
import io
from scipy.io.wavfile import write

print("Completed")

"""
Put noise in a file and saves the generated file 
1. reads the file
2. Extracts spectrogram from file
3. Generates a random file the same size of the file 
4. Back generated files to wav
5. Mixes the files
5. Saves the "noised" file in the random noise dir"""

files_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\data_toy"
generated_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\generated"
random_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\randomNoise"

FRAME_SIZE = 2048
HOP_SIZE = 512
#1. Lee archivos
files_names= os.listdir(files_dir)

#3. Load a to a numpy file each file
spec_files = [] #lista para guardar todos losa arrays en una matriz
for i, file in enumerate(files_names):
    numpy_file, sr_original = librosa.load(os.path.join(files_dir, file))
    spec_file = librosa.stft(numpy_file,  n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
    print("File read size (nr_freq_bins, nr_time_bins):", spec_file.shape )
    #generate the random noise
    noise_spec = np.random.randn(spec_file.shape[0], spec_file.shape[1])
    #Back to wav
    S = np.abs(spec_file)**2
    y_inv = librosa.griffinlim(S)
    noise = librosa.griffinlim(noise_spec)
    #Mixing the file
    noised_file = y_inv + noise
    #Save
    save_path = os.path.join(random_dir, str(i) + ".wav")
    sf.write(save_path, noised_file, sr_original)
    


print("Done!")