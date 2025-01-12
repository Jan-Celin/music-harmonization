import streamlit as st
import sounddevice as sd
import numpy as np
import wave
from io import BytesIO
import sys
import librosa
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

def extract_melody(audio_file):
    """
    Loads an audio file and extracts the melody using librosa.
    Args:
        audio_file (str): Path to the audio file.
    Returns:
        np.ndarray: The extracted melody (midi).
    """

    model_output, midi_data, note_events = predict('data/input.wav')

    return model_output, midi_data, note_events

audio_value = st.audio_input("Record the melody")

# Also allow the user to upload an audio file
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac"])

if audio_value:
    st.audio(audio_value)
"""
    # Save the audio to a file
    with open('data/input.wav', 'wb') as f:
        f.write(audio_value.read())
"""
# Extract the melody from the audio file
model_output, midi_data, note_events = extract_melody('data/input.wav')

# Display the melody
#st.line_chart(melody)
print(midi_data.instruments[0].notes)