from ProcessDataset import process_dataset, prepare_dataset
from TransformerModel import HarmonizerTransformer, collate_fn, midi_vocab_size, chord_vocab_sizes, SonataDataset, train_model, test_model, hyperparameter_search

from torch.utils.data import DataLoader
import torch

import datetime


device="cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
EPOCHS = 5


def main():
    print("Processing dataset started...")
    process_dataset('data/functional-harmony/BPS_FH_Dataset')
    print("Processing dataset completed.")

    # Load the dataset
    dataset_path = 'data/processed'
    train_test_split = 0.8
    data_train, data_test = prepare_dataset(dataset_path, train_test_split)

    print("Number of phrases in the train set:", len(data_train))
    print("Number of phrases in the test set:", len(data_test))

    dataset_train = SonataDataset(data_train)
    dataset_test = SonataDataset(data_test)

    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Perform hyperparameter search
    """
    print("Beginning hyperparameter search...\n")
    best_hyperparameters, best_accuracy = hyperparameter_search()
    print(f"Best hyperparameters: {best_hyperparameters}, Best accuracy: {best_accuracy}")
    d_model = best_hyperparameters['d_model']
    nhead = best_hyperparameters['nhead']
    num_encoder_layers = best_hyperparameters['num_encoder_layers']
    num_decoder_layers = best_hyperparameters['num_decoder_layers']
    learning_rate = best_hyperparameters['learning_rate']
    print("\nHyperparameter search ended.")
    """

    # Initialize the model
    model = HarmonizerTransformer(midi_vocab_size, chord_vocab_sizes)

    # Train the model
    trained_model = train_model(
        model=model,
        dataloader=dataloader_train,
        chord_vocab_sizes=chord_vocab_sizes,
        num_epochs=EPOCHS,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test the model accuracy
    accuracy = test_model(trained_model, dataloader_test, chord_vocab_sizes)
    print(f"Test set accuracy: {accuracy}")

    # Save the model
    torch.save(trained_model.state_dict(), f'saved_models/harmonizer_transformer_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pth')


if __name__ == '__main__':
    main()