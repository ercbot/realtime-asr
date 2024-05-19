import numpy as np

def convert_to_single_channel(audio_data):
    # Check if the audio data is already single-channel
    if len(audio_data.shape) == 1:
        return audio_data
    elif len(audio_data.shape) == 2 and audio_data.shape[1] == 2:
        single_channel_audio = np.mean(audio_data, axis=1)
        return single_channel_audio
    else:
        raise ValueError(f"Unexpected audio data shape: {audio_data.shape}")


def rms_amplitude(audio_data):
    # Calculate the root mean square (RMS) amplitude of the audio data
    rms = np.sqrt(np.mean(audio_data ** 2))
    return rms