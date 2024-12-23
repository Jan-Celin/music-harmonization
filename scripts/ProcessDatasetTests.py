from ProcessDataset import process_dataset, process_notes_df, process_chords_df
import os
import pandas as pd


def test_process_notes_df():
    print("process_notes_df() - Running tests...")

    notes_df = pd.read_csv('data/functional-harmony/BPS_FH_Dataset/1/notes.csv', header=None)
    notes_df_processed = process_notes_df(notes_df)

    assert len(notes_df) >= len(notes_df_processed)
    print("process_notes_df() - Test 1 passed: Notes dataframe length makes sense.")

    assert notes_df_processed['onset_time'].duplicated().any() == False
    print("process_notes_df() - Test 2 passed: No duplicate onset times.")

    print("process_notes_df() - All tests passed successfully.")

def test_process_chords_df():
    print("process_chords_df() - Running tests...")

    chords_df = pd.read_excel('data/functional-harmony/BPS_FH_Dataset/1/chords.xlsx', header=None)
    chords_df_processed = process_chords_df(chords_df)

    assert chords_df_processed['key'].dtype == 'int64'
    print("process_chords_df() - Test 1 passed: 'key' column processed successfully.")

    assert chords_df_processed['degree'].dtype == 'int64'
    print("process_chords_df() - Test 2 passed: 'degree' column processed successfully.")

    assert chords_df_processed['quality'].dtype == 'int64'
    print("process_chords_df() - Test 3 passed: 'quality' column processed successfully.")

    assert chords_df_processed['inversion'].dtype == 'int64'
    print("process_chords_df() - Test 4 passed: 'inversion' column processed successfully.")

    assert chords_df_processed['time'].duplicated().any() == False
    print("process_chords_df() - Test 5 passed: No duplicate time values.")

    assert len(chords_df_processed['time'].unique()) == max(chords_df_processed['time']) - min(chords_df_processed['time']) + 1
    print("process_chords_df() - Test 6 passed: No missing time values.")

    print("process_chords_df() - All tests passed successfully.")

def test_process_dataset(path):
    print("process_dataset() - Running tests...")

    dataset_path = path
    process_dataset(dataset_path)

    for dir in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, dir)):
            continue
        assert os.path.exists('data/processed/' + dir) == True 
    print("process_dataset() - Test 1: Files created successfully in 'data/processed' directory.")

    assert os.path.exists('data/processed/1/notes.csv') == True
    print("process_dataset() - Test 2: 'notes.csv' file created successfully.")

    assert os.path.exists('data/processed/1/chords.csv') == True
    print("process_dataset() - Test 3: 'chords.csv' file created successfully.")

    assert os.path.exists('data/processed/1/phrases.xlsx') == True
    print("process_dataset() - Test 4: 'phrases.xlsx' file copied successfully.")

    print("process_dataset() - All tests passed successfully.")


def main():
    print("Running tests for ProcessDataset.py...\n")

    test_process_notes_df()
    test_process_chords_df()
    test_process_dataset('data/functional-harmony/BPS_FH_Dataset')

    print("\nAll tests passed successfully!")


if __name__ == '__main__':
    main()