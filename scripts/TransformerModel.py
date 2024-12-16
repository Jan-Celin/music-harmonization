import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Transformer

import os
import math

from ProcessDataset import chord_key_to_int, scale_degree_to_int, chord_quality_to_int, process_dataset


midi_vocab_size = 128
chord_vocab_sizes = {
    'key': len(chord_key_to_int),
    'degree': len(scale_degree_to_int),
    'quality': len(chord_quality_to_int),
    'inversion': 4    # 4 possible chord inversions (Root, 1st, 2nd, 3rd)
}

class SinusoidalPositionalEncoding(nn.Module):
    # This StackOverflow answer was used as reference for this class: https://stackoverflow.com/a/77445896/21102779
    def __init__(self, d_model, max_time=10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_time = max_time

    def forward(self, onset_times):
        batch_size, seq_len = onset_times.size()

        positions = onset_times.unsqueeze(-1)

        div_term = torch.exp( torch.arange(0, self.d_model, 2, dtype=torch.float32) *  (-math.log(self.max_time) / self.d_model))

        embeddings = torch.zeros(batch_size, seq_len, self.d_model)
        embeddings[:, :, 0::2] = torch.sin(positions * div_term)
        embeddings[:, :, 1::2] = torch.cos(positions * div_term)

        return embeddings

class HarmonizerTransformer(nn.Module):
    def __init__(self, 
                 midi_vocab_size, 
                 chord_vocab_sizes, 
                 d_model=256, 
                 dropout=0.1,
                 nhead=8, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6):
        super(HarmonizerTransformer, self).__init__()

        self.d_model = d_model

        # Define the embedding for midi notes
        self.midi_embedding = nn.Embedding(midi_vocab_size, d_model)

        # Define separate embeddings for each chord property
        self.key_embedding = nn.Embedding(chord_vocab_sizes['key'], d_model)
        self.degree_embedding = nn.Embedding(chord_vocab_sizes['degree'], d_model)
        self.quality_embedding = nn.Embedding(chord_vocab_sizes['quality'], d_model)
        self.inversion_embedding = nn.Embedding(chord_vocab_sizes['inversion'], d_model)

        # Define the positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)

        # Define the transformer architecture
        self.transformer = Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout=dropout
        )

        # Define the output heads. The transformer will output four values:
        # - chord key
        # - chord degree
        # - chord quality
        # - chord inversion
        self.key_head = nn.Linear(d_model, chord_vocab_sizes['key'])
        self.degree_head = nn.Linear(d_model, chord_vocab_sizes['degree'])
        self.quality_head = nn.Linear(d_model, chord_vocab_sizes['quality'])
        self.inversion_head = nn.Linear(d_model, chord_vocab_sizes['inversion'])

    def forward(self, src_notes, src_onset_times, tgt_chords, tgt_onset_times):
        # Embed the midi notes
        src_notes_embeddings = self.midi_embedding(src_notes)
        src_positional_encodings = self.positional_encoding(src_onset_times)
        src = src_notes_embeddings + src_positional_encodings

        # Embed the target chords (each separately)
        key_emb = self.key_embedding(tgt_chords[:, :, 0])
        degree_emb = self.degree_embedding(tgt_chords[:, :, 1])
        quality_emb = self.quality_embedding(tgt_chords[:, :, 2])
        inversion_emb = self.inversion_embedding(tgt_chords[:, :, 3])

        # Combine the chord embeddings
        tgt_emb = key_emb + degree_emb + quality_emb + inversion_emb
        tgt_onset_emb = self.positional_encoding(tgt_onset_times)
        tgt = tgt_emb + tgt_onset_emb

        # Permute the values to have the shape (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Pass the midi embeddings through the encoder to get the memory
        memory = self.transformer.encoder(src)

        # Pass the target embeddings through the decoder to get the transformer's output
        output = self.transformer.decoder(tgt, memory)

        # Get the output for each head
        key = self.key_head(output)
        degree = self.degree_head(output)
        quality = self.quality_head(output)
        inversion = self.inversion_head(output)

        return key, degree, quality, inversion


def main():
    model = HarmonizerTransformer(midi_vocab_size, chord_vocab_sizes)

    # Load sample data
    df_notes = pd.read_csv("data/processed/1/notes.csv")
    df_chords = pd.read_csv("data/processed/1/chords.csv")

    # Prepare the input tensors
    src_notes = torch.tensor(df_notes["midi_note"].tolist(), dtype=torch.long).unsqueeze(0)
    src_onset_times = torch.tensor(df_notes["onset_time"].tolist(), dtype=torch.long).unsqueeze(0)

    # Prepare the target tensors
    tgt_chords = torch.tensor(df_chords[["key", "degree", "quality", "inversion"]].values).unsqueeze(0)
    tgt_onset_times = torch.tensor(df_chords["time"].values, dtype=torch.long).unsqueeze(0)

    # Forward pass
    key, degree, quality, inversion = model(src_notes, src_onset_times, tgt_chords, tgt_onset_times)

    # Get the predicted values
    key_pred = key.argmax(dim=-1)
    degree_pred = degree.argmax(dim=-1)
    quality_pred = quality.argmax(dim=-1)
    inversion_pred = inversion.argmax(dim=-1)

    print(key_pred)
    print(degree_pred)
    print(quality_pred)
    print(inversion_pred)
    

if __name__ == "__main__":
    main()