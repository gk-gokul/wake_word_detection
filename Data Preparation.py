import os
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import queue


def record_audio_and_save(save_path, n_times=100):
    """
    This function will record your voice `n_times` for 2 seconds each time.

    Parameters
    ----------
    save_path: str
        Directory where to save the wav files.
    n_times: int, default=100
        Number of recordings to make.
    """

    fs = 44100
    seconds = 2

    for i in range(n_times):
        print(f"Recording your voice {i+1}/{n_times}")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(f"{save_path}/voice_{i}.wav", fs, myrecording)


def record_background_sound(save_path, n_times=100):
    """
    This function will record background sounds `n_times` for 2 seconds each time.

    Parameters
    ----------
    save_path: str
        Directory where to save the wav files.
    n_times: int, default=100
        Number of recordings to make.
    """

    fs = 44100
    seconds = 2

    for i in range(n_times):
        print(f"Recording background sound {i+1}/{n_times}")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(f"{save_path}/background_{i}.wav", fs, myrecording)


# Create directories for saving files
os.makedirs("audio_data/voice", exist_ok=True)
os.makedirs("audio_data/background", exist_ok=True)

# Start recording voice
print("Recording your voice:\n")
record_audio_and_save("audio_data/voice")

# Start recording background sounds
print("Recording background sounds:\n")
record_background_sound("audio_data/background")
