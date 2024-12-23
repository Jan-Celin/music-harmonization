from TransformerModel import HarmonizerTransformer, SinusoidalPositionalEncoding, midi_vocab_size, chord_vocab_sizes, SonataDataset, collate_fn, train_model, test_model
from ProcessDataset import prepare_dataset

import torch
from torch.utils.data import Dataset, DataLoader

def test_SinusoidalPositionalEncoding():
    print("SinusoidalPositionalEncoding() - Running tests...")

    positional_encoding = SinusoidalPositionalEncoding(512)
    onset_times = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    onset_times = onset_times.unsqueeze(0)
    encoding = positional_encoding(onset_times)
    assert encoding.shape == (1, 10, 512)
    print("SinusoidalPositionalEncoding() - Test 1 passed: Correct shape.")

    print("SinusoidalPositionalEncoding() - All tests passed successfully.")

def test_HarmonizerTransformer():
    print("HarmonizerTransformer() - Running tests...")

    dataset_path = 'data/processed'
    train_test_split = 0.1
    data_train, data_test = prepare_dataset(dataset_path, train_test_split)

    dataset_train = SonataDataset(data_train)
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = HarmonizerTransformer(midi_vocab_size, chord_vocab_sizes)

    trained_model = train_model(
        model=model,
        dataloader=dataloader_train,
        chord_vocab_sizes=chord_vocab_sizes,
        num_epochs=1,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("HarmonizerTransformer() - Test 1 passed: Model training successfully finished (on small subset of data).")

    accuracy = test_model(trained_model, dataloader_train, chord_vocab_sizes)
    
    print("HarmonizerTransformer() - Test 1 passed: Model testing (inference) successfully finished (on small subset of data).")

    print("HarmonizerTransformer() - All tests passed successfully.")

def main():
    print("Running tests for TransformerModel.py...\n")

    test_SinusoidalPositionalEncoding()
    test_HarmonizerTransformer()

    print("\nAll tests passed successfully!")


if __name__ == '__main__':
    main()