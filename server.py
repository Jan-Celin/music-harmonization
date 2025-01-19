import json
from flask import Flask, request, render_template_string
import torch
from scripts.TransformerModel import HarmonizerTransformer, midi_vocab_size, chord_vocab_sizes 

app = Flask(__name__)

# Set up the model
model = HarmonizerTransformer(midi_vocab_size=midi_vocab_size, chord_vocab_sizes=chord_vocab_sizes)
model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=torch.device('cpu')))


@app.route("/")
def index():
    return render_template_string("To get the melody's harmonies, please send a GET request to /harmonize")


@app.route("/harmonize", methods=['POST'])
def harmonize():
    """
    Method that receives a melody and returns its harmonies.

    Example payload:
    {
        "note_events": [
            {"onset": 0.0, "pitch": 60.0},
            {"onset": 0.5, "pitch": 62.0},
            ...
        ]
    }

    Returns:
    Dict[str, List[int]]: The harmonized melody.

    Example response:
    {
        "chord_keys": [0, 2, ...],
        "chord_degrees": [0, 2, ...],
        "chord_qualities": [0, 2, ...],
        "chord_inversions": [0, 2, ...]
    }
    """
    data = request.get_json()

    # Extract the melody from the payload
    note_events = data["note_events"]
    onsets = [note["onset"] for note in note_events]
    pitches = [note["pitch"] for note in note_events]

    # Harmonize the melody
    harmonized_pitches = model.harmonize(onsets, pitches)
    print(harmonized_pitches)
    
    chord_keys = harmonized_pitches[0, :, 0].tolist()
    chord_degrees = harmonized_pitches[0, :, 1].tolist()
    chord_qualities = harmonized_pitches[0, :, 2].tolist()
    chord_inversions = harmonized_pitches[0, :, 3].tolist()

    # Return the harmonized melody
    response = {
        "chord_keys": chord_keys,
        "chord_degrees": chord_degrees,
        "chord_qualities": chord_qualities,
        "chord_inversions": chord_inversions,
    }
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1111, debug=True)