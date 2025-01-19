import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import numpy as np

from tqdm import tqdm

import math

from .ProcessDataset import chord_key_to_int, scale_degree_to_int, chord_quality_to_int, process_dataset, prepare_dataset


device="cpu"
print("Device:", device)

midi_vocab_size = 128
chord_vocab_sizes = {
    'key': len(chord_key_to_int)+1,
    'degree': len(scale_degree_to_int)+1,
    'quality': len(chord_quality_to_int)+1,
    'inversion': 4+1    # 4 possible chord inversions (Root, 1st, 2nd, 3rd)
}

class SinusoidalPositionalEncoding(nn.Module):
    """
    Class used to calculate positional embeddings of music events by their onset times.
    """

    # This StackOverflow answer was used as reference for this class: https://stackoverflow.com/a/77445896/21102779
    def __init__(self, d_model, max_time=10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_time = max_time

    def forward(self, onset_times):
        batch_size, seq_len = onset_times.size()

        positions = onset_times.unsqueeze(-1).to(device)

        div_term = torch.exp( torch.arange(0, self.d_model, 2, dtype=torch.float32) *  (-math.log(self.max_time) / self.d_model)).to(device)

        embeddings = torch.zeros(batch_size, seq_len, self.d_model)
        embeddings[:, :, 0::2] = torch.sin(positions * div_term)
        embeddings[:, :, 1::2] = torch.cos(positions * div_term)

        return embeddings


class SonataDataset(Dataset):
    """
    Class which wraps the torch.utils.data.Dataset class. 
    Defines methods for initializing and getting elements of the sonata dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(batch):
    """
    Function to pad the sequences in a batch to the same length.

    Args:
        batch (list): List of dictionaries

    Returns:
        dict: Dictionary with the padded sequences
    """

    src_notes = [item['src_notes'] for item in batch]
    src_onsets = [item['src_onsets'] for item in batch]
    tgt_chords = [item['tgt_chords'] for item in batch]
    tgt_onsets = [item['tgt_onsets'] for item in batch]

    # Pad sequences to the same length in each batch
    src_notes_padded = pad_sequence(src_notes, batch_first=True, padding_value=0)
    src_onsets_padded = pad_sequence(src_onsets, batch_first=True, padding_value=0)
    tgt_chords_padded = pad_sequence(tgt_chords, batch_first=True, padding_value=0)
    tgt_onsets_padded = pad_sequence(tgt_onsets, batch_first=True, padding_value=0)

    return {
        'src_notes': src_notes_padded,
        'src_onsets': src_onsets_padded,
        'tgt_chords': tgt_chords_padded,
        'tgt_onsets': tgt_onsets_padded
    }


class HarmonizerTransformer(nn.Module):
    """
    Class which defines the transformer model for harmonization.
    """
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

    def forward(self, src_notes, src_onset_times, tgt_chords, tgt_onset_times, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Embed the midi notes
        src_notes_embeddings = self.midi_embedding(src_notes.cpu().long()).to(device)
        src_positional_encodings = self.positional_encoding(src_onset_times).to(device)
        src = src_notes_embeddings + src_positional_encodings

        # Embed the target chords (each separately)
        key_emb = self.key_embedding(tgt_chords[:, :, 0].to(device))
        degree_emb = self.degree_embedding(tgt_chords[:, :, 1].to(device))
        quality_emb = self.quality_embedding(tgt_chords[:, :, 2].to(device))
        inversion_emb = self.inversion_embedding(tgt_chords[:, :, 3].to(device))

        # Combine the chord embeddings
        tgt_emb = key_emb + degree_emb + quality_emb + inversion_emb
        tgt_emb = tgt_emb.to(device)
        tgt_onset_emb = self.positional_encoding(tgt_onset_times).to(device)
        tgt = tgt_emb + tgt_onset_emb

        # Permute the values to have the shape (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Pass the target embeddings through the decoder to get the transformer's output
        output = self.transformer(src.to(device), tgt.to(device), tgt_mask=tgt_mask)

        # Get the output for each head
        key = self.key_head(output)
        degree = self.degree_head(output)
        quality = self.quality_head(output)
        inversion = self.inversion_head(output)

        return key, degree, quality, inversion

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def harmonize(self, onsets, pitches):
        """
        Function to harmonize a melody given the onset times and pitches.

        Args:
            onsets (list): List of onset times
            pitches (list): List of pitches

        Returns:
            list: dictionary with the harmony values (key, degree, quality, inversion)
        """

        # Convert the input melody to tensors
        onsets = torch.tensor(onsets).unsqueeze(0).to(device)
        pitches = torch.tensor(pitches).unsqueeze(0).to(device)

        # Generate the positional encodings
        key_pred = [0 for i in range(onsets.size(1))]
        degree_pred = [0 for i in range(onsets.size(1))]
        quality_pred = [0 for i in range(onsets.size(1))]
        inversion_pred = [0 for i in range(onsets.size(1))]

        # Combine tgt_key_pred, tgt_degree_pred, tgt_quality_pred, tgt_inversion_pred into a tensor of shape (batch_size, seq_len, 4) and convert all zeros to int
        tgt_chords_pred = torch.zeros((1, onsets.size(1), 4)).to(device)
        tgt_chords_pred = tgt_chords_pred.long()

        # Autoregressive inference
        for i in range(onsets.size(1)+1):
            key_pred, degree_pred, quality_pred, inversion_pred = self(onsets, pitches, tgt_chords_pred, onsets)

            key_pred = torch.from_numpy(np.argmax(key_pred.detach().cpu().numpy(), axis=-1)).permute(1, 0).to(device)
            degree_pred = torch.from_numpy(np.argmax(degree_pred.detach().cpu().numpy(), axis=-1)).permute(1, 0).to(device)
            quality_pred = torch.from_numpy(np.argmax(quality_pred.detach().cpu().numpy(), axis=-1)).permute(1, 0).to(device)
            inversion_pred = torch.from_numpy(np.argmax(inversion_pred.detach().cpu().numpy(), axis=-1)).permute(1, 0).to(device)

            # Combine tgt_key_pred, tgt_degree_pred, tgt_quality_pred, tgt_inversion_pred into a tensor of shape (batch_size, seq_len, 4)
            tgt_chords_pred[:, :, 0] = key_pred
            tgt_chords_pred[:, :, 1] = degree_pred
            tgt_chords_pred[:, :, 2] = quality_pred
            tgt_chords_pred[:, :, 3] = inversion_pred

        harmonies = []
        for i in range(onsets.size(1)):
            harmony = {
                'onset': onsets[0, i].item(),
                'key': tgt_chords_pred[0, i, 0].item(),
                'degree': tgt_chords_pred[0, i, 1].item(),
                'quality': tgt_chords_pred[0, i, 2].item(),
                'inversion': tgt_chords_pred[0, i, 3].item()
            }
            harmonies.append(harmony)
        return tgt_chords_pred

def train_model(model, dataloader, chord_vocab_sizes, num_epochs=10, learning_rate=0.001, device='cuda'):
    """
    Function to train the transformer model.

    Args:
        model (HarmonizerTransformer): The transformer model
        dataloader (DataLoader): DataLoader object with the dataset
        chord_vocab_sizes (dict): Dictionary with the sizes of the chord vocabularies
        num_epochs (int): Number of epochs (default: 10)
        learning_rate (float): Learning rate (default: 0.001)
        device (str): Device to use ('cpu' or 'cuda') (default: 'cuda')

    Returns:
        HarmonizerTransformer: Trained transformer model
    """

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss functions for each output head
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            src_notes = batch['src_notes'].to(device)
            src_onsets = batch['src_onsets'].to(device)
            tgt_chords = batch['tgt_chords'].to(device)
            tgt_onsets = batch['tgt_onsets'].to(device)

            tgt_chords_in = tgt_chords[:, :-1, :]
            tgt_chords_out = tgt_chords[:, 1:, :]

            tgt_onsets_in = tgt_onsets[:, :-1]
            tgt_onsets_out = tgt_onsets[:, 1:] # Redundant

            tgt_key = tgt_chords_out[:, :, 0]
            tgt_degree = tgt_chords_out[:, :, 1]
            tgt_quality = tgt_chords_out[:, :, 2]
            tgt_inversion = tgt_chords_out[:, :, 3]

            seq_len = tgt_chords_in.size(1)
            tgt_mask = model.get_tgt_mask(seq_len).to(device)

            # Forward pass
            key_pred, degree_pred, quality_pred, inversion_pred = model(src_notes, src_onsets, tgt_chords_in, tgt_onsets_in, tgt_mask=tgt_mask)

            # Reshape predictions and targets to compute the loss
            key_pred = key_pred.view(-1, chord_vocab_sizes['key'])  # (batch_size * seq_len, vocab_size)
            degree_pred = degree_pred.view(-1, chord_vocab_sizes['degree'])
            quality_pred = quality_pred.view(-1, chord_vocab_sizes['quality'])
            inversion_pred = inversion_pred.view(-1, chord_vocab_sizes['inversion'])

            tgt_key = tgt_key.reshape(-1)  # (batch_size * seq_len)
            tgt_degree = tgt_degree.reshape(-1)
            tgt_quality = tgt_quality.reshape(-1)
            tgt_inversion = tgt_inversion.reshape(-1)

            # Compute the loss for each output head
            loss_key = criterion(key_pred, tgt_key)
            loss_degree = criterion(degree_pred, tgt_degree)
            loss_quality = criterion(quality_pred, tgt_quality)
            loss_inversion = criterion(inversion_pred, tgt_inversion)

            total_loss = loss_key + loss_degree + loss_quality + loss_inversion

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

    return model

def test_model(model, dataloader, chord_vocab_sizes):
    """
    Function to test the transformer model.

    Args:
        model (HarmonizerTransformer): The transformer model
        dataloader (DataLoader): DataLoader object with the dataset
        chord_vocab_sizes (dict): Dictionary with the sizes of the chord vocabularies

    Returns:
        float: Accuracy of the model
    """
    model.eval()
    correct_predictions_key = 0
    correct_predictions_degree = 0
    correct_predictions_quality = 0
    correct_predictions_inversion = 0
    total_predictions = 0

    for batch in tqdm(dataloader, desc="Testing"):
        src_notes = batch['src_notes'].to("cuda" if torch.cuda.is_available() else "cpu")
        src_onsets = batch['src_onsets'].to("cuda" if torch.cuda.is_available() else "cpu")
        tgt_chords_true = batch['tgt_chords'].to("cuda" if torch.cuda.is_available() else "cpu")
        tgt_onsets_true = batch['tgt_onsets'].to("cuda" if torch.cuda.is_available() else "cpu")

        tgt_key_true = tgt_chords_true[:, :, 0].to(device)
        tgt_degree_true = tgt_chords_true[:, :, 1].to(device)
        tgt_quality_true = tgt_chords_true[:, :, 2].to(device)
        tgt_inversion_true = tgt_chords_true[:, :, 3].to(device)

        key_pred = [0 for i in range(src_onsets.size(1)+1)]
        degree_pred = [0 for i in range(src_onsets.size(1)+1)]
        quality_pred = [0 for i in range(src_onsets.size(1)+1)]
        inversion_pred = [0 for i in range(src_onsets.size(1)+1)]
        # Combine tgt_key_pred, tgt_degree_pred, tgt_quality_pred, tgt_inversion_pred into a tensor of shape (batch_size, seq_len, 4) and convert all zeros to int
        tgt_chords_pred = torch.zeros((tgt_chords_true.size(0), tgt_chords_true.size(1), 4)).to("cuda" if torch.cuda.is_available() else "cpu")
        tgt_chords_pred = tgt_chords_pred.long()

        # Autoregressive inference
        for i in range(src_onsets.size(1)+1):
            key_pred, degree_pred, quality_pred, inversion_pred = model(src_notes, src_onsets, tgt_chords_pred, tgt_onsets_true)

            key_pred = torch.from_numpy(np.argmax(key_pred.detach().cpu().numpy(), axis=-1)).permute(1, 0).to(device)
            degree_pred = torch.from_numpy(np.argmax(degree_pred.detach().cpu().numpy(), axis=-1)).permute(1, 0).to(device)
            quality_pred = torch.from_numpy(np.argmax(quality_pred.detach().cpu().numpy(), axis=-1)).permute(1, 0).to(device)
            inversion_pred = torch.from_numpy(np.argmax(inversion_pred.detach().cpu().numpy(), axis=-1)).permute(1, 0).to(device)

            # Combine tgt_key_pred, tgt_degree_pred, tgt_quality_pred, tgt_inversion_pred into a tensor of shape (batch_size, seq_len, 4)
            tgt_chords_pred[:, :, 0] = key_pred.to(device)
            tgt_chords_pred[:, :, 1] = degree_pred.to(device)
            tgt_chords_pred[:, :, 2] = quality_pred.to(device)
            tgt_chords_pred[:, :, 3] = inversion_pred.to(device)

        # Compute the accuracy of each output head
        correct_predictions_key += torch.sum(tgt_key_true == key_pred).item()
        correct_predictions_degree += torch.sum(tgt_degree_true == degree_pred).item()
        correct_predictions_quality += torch.sum(tgt_quality_true == quality_pred).item()
        correct_predictions_inversion += torch.sum(tgt_inversion_true == inversion_pred).item()

        total_predictions += tgt_key_true.numel()

    # Compute the accuracy
    accuracy_key = correct_predictions_key / total_predictions
    accuracy_degree = correct_predictions_degree / total_predictions
    accuracy_quality = correct_predictions_quality / total_predictions
    accuracy_inversion = correct_predictions_inversion / total_predictions

    print(f"Accuracy key: {accuracy_key}, Accuracy degree: {accuracy_degree}, Accuracy quality: {accuracy_quality}, Accuracy inversion: {accuracy_inversion}")

    return (accuracy_key + accuracy_degree + accuracy_quality + accuracy_inversion) / 4


def hyperparameter_search():
    """
    Function which contains code for hyperparameter search.
    """

    # Define the hyperparameters to search
    hyperparameters = {
        'd_model': [128, 256, 512],
        'nhead': [4, 8, 16],
        'num_encoder_layers': [4, 6, 8],
        'num_decoder_layers': [4, 6, 8],
        'learning_rate': [0.01, 0.001, 0.0005, 0.0001, 0.00005]
    }

    # Load the dataset
    dataset_path = 'data/processed'
    train_test_split = 0.8
    data_train, data_test = prepare_dataset(dataset_path, train_test_split)

    dataset_train = SonataDataset(data_train)
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Perform the hyperparameter search
    best_accuracy = 0
    best_hyperparameters = None

    for d_model in hyperparameters['d_model']:
        for nhead in hyperparameters['nhead']:
            for num_encoder_layers in hyperparameters['num_encoder_layers']:
                for num_decoder_layers in hyperparameters['num_decoder_layers']:
                    for learning_rate in hyperparameters['learning_rate']:
                        model = HarmonizerTransformer(midi_vocab_size, chord_vocab_sizes, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
                        trained_model = train_model(
                            model=model,
                            dataloader=dataloader_train,
                            chord_vocab_sizes=chord_vocab_sizes,
                            num_epochs=5,
                            learning_rate=learning_rate,
                            device="cuda" if torch.cuda.is_available() else "cpu"
                        )

                        dataset_test = SonataDataset(data_test)
                        dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=collate_fn)

                        accuracy = test_model(trained_model, dataloader_test, chord_vocab_sizes)
                        print(f"Hyperparameters: d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}, num_decoder_layers={num_decoder_layers}, learning_rate={learning_rate}, Accuracy={accuracy}")

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_hyperparameters = {
                                'd_model': d_model,
                                'nhead': nhead,
                                'num_encoder_layers': num_encoder_layers,
                                'num_decoder_layers': num_decoder_layers,
                                'learning_rate': learning_rate
                            }

    return best_hyperparameters, best_accuracy


def main_original():
    # Load the dataset
    dataset_path = 'data/processed'
    train_test_split = 0.8
    data_train, data_test = prepare_dataset(dataset_path, train_test_split)

    dataset_train = SonataDataset(data_train)
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Perform hyperparameter search
    """
    best_hyperparameters, best_accuracy = hyperparameter_search()
    print(f"Best hyperparameters: {best_hyperparameters}, Best accuracy: {best_accuracy}")
    d_model = best_hyperparameters['d_model']
    nhead = best_hyperparameters['nhead']
    num_encoder_layers = best_hyperparameters['num_encoder_layers']
    num_decoder_layers = best_hyperparameters['num_decoder_layers']
    learning_rate = best_hyperparameters['learning_rate']
    """

    # Train the model
    model = HarmonizerTransformer(midi_vocab_size, chord_vocab_sizes)

    trained_model = train_model(
        model=model,
        dataloader=dataloader_train,
        chord_vocab_sizes=chord_vocab_sizes,
        num_epochs=1,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test the model accuracy
    dataset_test = SonataDataset(data_test)
    dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=collate_fn)

    accuracy = test_model(trained_model, dataloader_test, chord_vocab_sizes)
    print(f"Test set accuracy: {accuracy}")

    # Save the model
    #torch.save(trained_model.state_dict(), 'harmonizer_transformer.pth')


def main():
    model = HarmonizerTransformer(midi_vocab_size=midi_vocab_size, chord_vocab_sizes=chord_vocab_sizes)
    model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=torch.device('cpu')))
    
    dataset_path = 'data/processed'
    train_test_split = 0.8
    data_train, data_test = prepare_dataset(dataset_path, train_test_split)

    # Test the model
    dataset_test = SonataDataset(data_test)
    dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=collate_fn)

    accuracy = test_model(model, dataloader_test, chord_vocab_sizes)


if __name__ == "__main__":
    main()