
import librosa
import numpy as np
import os


def audio_files_to_numpy(files_dir, list_audio_files,
frame_length, hop_size,  min_duration):
    list_sound_array = []

    for file in list_audio_files:
        y, sr = librosa.load(os.path.join(files_dir, file))
        duration = librosa.get_duration(y, sr) 
        #if the array has the suitable duration append the
        # y separated in nb frames to list
        #if not let it pass
        if duration >= min_duration:
            list_sound_array.append(audio_to_frame_stack(y, frame_length,
            hop_size))
        else:
            print("The file {os.path.join(files_dir, file)} doesnt have the min duration")

       
    return np.vstack(list_sound_array) #genera una matrix de dimension (num_frames, frame_length)

       
   
def audio_to_frame_stack(file, frame_length,  hop_size):
    num_frames = len(y) - frame_length +1
    starlist = [file[i: i + frame_length] for i in range(0, num_frames, hop_size)]
    return starlist

