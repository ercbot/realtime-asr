import time

import gradio as gr
import numpy as np
import pandas as pd

import sounddevice as sd

from audio_processing import convert_to_single_channel
from transcription import load_transcriber



# Main Model
# model_id = "openai/whisper-large-v3"
# assistant_model_id = "distil-whisper/distil-large-v3"

model_id = "distil-whisper/distil-medium.en"
transcriber = load_transcriber(model_id)

start_time = time.time()


def transcribe(stream, new_chunk):
    sr, y = new_chunk

    y = y.astype(np.float32)
    y = convert_to_single_channel(y)

    # Normalize the audio
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


def audio_volume(df, new_chunk):
    elapsed_time = time.time() - start_time

    sr, y = new_chunk

    y = y.astype(np.float32)
    y = convert_to_single_channel(y)

    # Calculate the Volume
    volume = np.linalg.norm(y) * 10

    df_new = pd.DataFrame({'time': [elapsed_time], 'volume': [volume]})

    if df is not None:
        df = np.concatenate([df, df_new])
    else:
        df = y

    return df, df


# demo = gr.Interface(
#     transcribe,
#     ["state", gr.Audio(sources=["microphone"], streaming=True)],
#     ["state", "number"],
#     live=True,
# )

demo = gr.Interface(
    audio_volume,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "lineplot"],
    live=True,
)

demo.launch()