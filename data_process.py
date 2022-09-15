import librosa
import numpy as np
import os
import IPython.display as ipd
import soundfile as sf


"""
1. Lee los archivos del directorio
2. Informa sus nombres
3. load a nympy array de cada uno usando librosa
4. Reporta su sr_original y duración"""

files_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\data_toy"
generated_dir = r"C:\Users\malfaro\Desktop\mae_code\Denoiser\generated"
FRAME_SIZE = 2048
HOP_SIZE = 512
#1. Lee archivos
files_names= os.listdir(files_dir)

#2. Nombres
print ("Files in the folder:")
print("______________________")
for file in files_names:
    print(file)

#3. Load a to a numpy file each file
spec_files = [] #lista para guardar todos losa arrays en una matriz
for file in files_names:
    numpy_file, sr_original = librosa.load(os.path.join(files_dir, file))
    spec_file = librosa.stft(numpy_file,  n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
    print("Spec file size (nr_freq_bins, nr_time_bins):", spec_file.shape )
    #Back to wav
    S = np.abs(spec_file)**2
    y_inv = librosa.griffinlim(S)
    wav = sf.write("Test.wav", y_inv, sr_original)
    with open(os.path.join(generated_dir, wav), "w") as file1:
        file1.write()

    ##Guardar cada archivo de audio generado en una carpeta
    #Mezclar con noise y guardar los archivos noisy generados en una carpeta


print("Done.")