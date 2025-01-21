import streamlit as st
from basic_pitch.inference import predict
import requests

from music21 import pitch, chord, midi, stream, tempo

import io

import numpy as np
import pretty_midi
import requests
from scipy.io import wavfile
import os


chord_intervals = {
    0: [0, 4, 7],      # Major triad
    1: [0, 4, 7],      # Major triad
    2: [0, 3, 7],      # Minor triad
    3: [0, 4, 7, 11],  # Major seventh
    4: [0, 3, 7, 10],  # Minor seventh
    5: [0, 4, 6, 10],  # Dominant seventh
    6: [0, 4, 8],      # Augmented triad
    7: [0, 4, 8, 9],   # Augmented sixth
    8: [0, 3, 6],      # Diminished triad
    9: [0, 3, 6, 9],   # Diminished seventh
    10: [0, 3, 6, 10]  # Half-diminished seventh
}

chord_key_to_int = {
    1: 60,  2: 60,  3: 59,  4: 59,  5: 61,  6: 61,  7: 62,  8: 62,
    9: 61, 10: 61, 11: 63, 12: 63, 13: 64, 14: 64, 15: 63, 16: 63,
    17: 65, 18: 65, 19: 65, 20: 65, 21: 64, 22: 64, 23: 66, 24: 66,
    25: 67, 26: 67, 27: 66, 28: 66, 29: 68, 30: 68, 31: 69, 32: 69,
    33: 68, 34: 68, 35: 70, 36: 70, 37: 71, 38: 71, 39: 70, 40: 70,
    41: 72, 42: 72, 0: 60
}


def load_local_midi(file_path: str) -> bytes:
    """
    Load a local MIDI file from the specified file path.

    Args:
        file_path (str): The path to the local MIDI file.

    Returns:
        bytes: The binary content of the MIDI file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    with open(file_path, "rb") as midi_file:
        midi_data = midi_file.read()
    
    return midi_data


def create_chord(chord_key: int, scale_degree: int, quality: int, inversion: int = 0):
    # Calculate the root pitch from chord key and scale degree
    root_pitch = chord_key_to_int[chord_key]
    degree_offset = scale_degree
    root_pitch += degree_offset - 1

    # Generate the chord notes based on intervals
    intervals = chord_intervals[quality]
    chord_notes = [pitch.Pitch(midi=root_pitch + interval) for interval in intervals]
    chord_tone = chord.Chord(chord_notes)
    # Apply inversion
    chord_tone.inversion(inversion)
    return chord_tone

def create_midi(chords, durations, output_path):
    # Create a MIDI stream
    midi_stream = stream.Stream()
    tempo_marking = tempo.MetronomeMark(number=60)
    midi_stream.append(tempo_marking)

    # Add chords to the stream
    for c, duration in zip(chords, durations):
        chord_tone = c
        chord_tone.duration.quarterLength = duration
        midi_stream.append(chord_tone)

    # Write the MIDI stream to a file
    midi_stream.write("midi", output_path)

def play_midi(midi_path: str):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    audio_data = midi_data.fluidsynth()
    audio_data = np.int16(
        audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
    )

    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, 44100, audio_data)

    st.write("### The harmonized chords are ready!")
    st.audio(virtualfile)



        

def extract_melody(audio_file):
    """
    Loads an audio file and extracts the melody using librosa.
    Args:
        audio_file (str): Path to the audio file.
    Returns:
        np.ndarray: The extracted melody (midi).
    """

    model_output, midi_data, note_events = predict("data/input.wav")

    return model_output, midi_data, note_events

st.title("Melody Harmonizer")

st.markdown("### Please input your melody")
audio_value = st.audio_input("Record the melody")

st.markdown("""
    <div style="display: flex; align-items: center; text-align: center; justify-content: center;">
        <hr style="flex-grow: 1; border: 0; border-top: 1px solid #ccc; margin: 0; opacity: 0.4;">
        <span style="padding: 0 10px; font-weight: bold; color: #ccc;">or</span>
        <hr style="flex-grow: 1; border: 0; border-top: 1px solid #ccc; margin: 0; opacity: 0.4;">
    </div>
""", unsafe_allow_html=True)

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac"])

st.markdown("---")

if audio_value:
    # Save the audio file
    with open("data/input.wav", "wb") as f:
        f.write(audio_value.getbuffer())

    st.write("Your recorded melody:")
    st.audio(audio_value)

    st.markdown("---")

    with st.spinner(f"Creating the harmonies..."):
        # Extract the melody from the audio file
        model_output, midi_data, note_events = extract_melody("data/input.wav")

        # Send a request to the harmonize endpoint
        note_events = [{"onset": float(note.start), "pitch": float(note.pitch)} for note in midi_data.instruments[0].notes]
        payload = {
            "note_events": note_events,
        }
        response = requests.post("http://localhost:1111/harmonize", json=payload).json()

        chords = []
        for key, degree, quality, inversion in zip(response["chord_keys"], response["chord_degrees"], response["chord_qualities"], response["chord_inversions"]):
            chord_i = create_chord(key, degree, quality, inversion)
            chords.append(chord_i)

        # Create a MIDI file with the harmonized chords
        lengths = []
        for note in midi_data.instruments[0].notes:
            lengths.append(note.end - note.start)
        lengths[0] = midi_data.instruments[0].notes[0].end
        lengths[-1] = 2.0

        midi_path = "data/harmonized.mid"
        create_midi(chords, lengths, midi_path)

        # Play the harmonized chords
        try:
            play_midi(midi_path)
        except Exception as e:
            st.write("An error occurred while playing the harmonized chords. The issue is most likely due to the fact that you do not have FluidSynth installed in your system PATH. You can still download the MIDI file with the harmonized chords below.")
        st.write("Download the midi file with the harmonized chords:")
        st.download_button("Download harmonized chords", midi_path, "chords.mid", "audio/midi")

        st.markdown("---")
        st.write("### Chords:")
        begin_times = [0.00]
        for note in midi_data.instruments[0].notes:
            begin_times.append(float(str(round(note.start, 2))))

        for chord_i, begin_time in zip(chords, begin_times):
            chord_notes = ""
            for p in chord_i.pitches:
                chord_notes += p.nameWithOctave + "   "
            st.write(f"Chord for time {begin_time}s: {chord_notes}")
    