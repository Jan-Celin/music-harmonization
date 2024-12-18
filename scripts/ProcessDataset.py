import pandas as pd
import os
import torch
import shutil


chord_quality_to_int = {
    'M': 0,
    'm': 1,
    'M7': 2,
    'm7': 3,
    'D7': 4,
    'a': 5,
    'a6': 6,
    'd': 7,
    'd7': 8,
    'h7': 9
}

scale_degree_to_int = {
    '1': 1,
    '+1': 2,
    '-2': 2,
    '2': 3,
    '+2': 4,
    '-3': 4,
    '3': 5,
    '4': 6,
    '+4': 7,
    '-5': 7,
    '5': 8,
    '+5': 9,
    '-6': 9,
    '6': 10,
    '+6': 11,
    '-7': 11,
    '7': 12
}

chord_key_to_int = {
    'C': 1,
    'c': 2,
    'C-': 3,
    'c-': 4,
    'C+': 5,
    'c+': 6,
    'D': 7,
    'd': 8,
    'D-': 9,
    'd-': 10,
    'D+': 11,
    'd+': 12,
    'E': 13,
    'e': 14,
    'E-': 15,
    'e-': 16,
    'E+': 17,
    'e+': 18,
    'F': 19,
    'f': 20,
    'F-': 21,
    'f-': 22,
    'F+': 23,
    'f+': 24,
    'G': 25,
    'g': 26,
    'G-': 27,
    'g-': 28,
    'G+': 29,
    'g+': 30,
    'A': 31,
    'a': 32,
    'A-': 33,
    'a-': 34,
    'A+': 35,
    'a+': 36,
    'B': 37,
    'b': 38,
    'B-': 39,
    'b-': 40,
    'B+': 41,
    'b+': 42
}

def process_notes_df(df_notes):
    """
    Process the notes dataframe. The function gives meaningful names to the dataframe's columns and
    filters the notes to keep only the note with thehighest pitch played at each onset time.

    Args:
        df_notes (Pandas.Dataframe): Source notes dataframe.

    Returns:
        Pandas.Dataframe: Processed notes dataframe.
    """

    # Rename columns to give them meaningful names
    df_notes = df_notes.rename(columns={
        0: 'onset_time',
        1: 'midi_note',
        2: 'morphetic_pitch_number',
        3: 'duration',
        4: 'staff_number',
        5: 'measure'
    })

    # Clear the dataset to keep only the highest pitch note played at each onset time
    df_notes = df_notes.sort_values(by=['onset_time', 'midi_note'], ascending=[True, False])
    monophonic_melody = []
    current_onset = None

    for _, row in df_notes.iterrows():
        onset_time = row['onset_time']
        midi_note = row['midi_note']
        
        # Since the dataset is sorted, if there are multiple notes played at the same time, we only keep the first one (the one with the highest pitch).
        if onset_time != current_onset:
            monophonic_melody.append(row)
            current_onset = onset_time

    df_notes_clean = pd.DataFrame(monophonic_melody).reset_index(drop=True)
    return df_notes_clean

def process_chords_df(df_chords):
    """
    Process the chords dataframe. The function gives meaningful names to the dataframe's columns and
    expands the dataset to have one row per each integer onset-time, with the chord played at that time.

    Args:
        df_chords (Pandas.Dataframe): Source chords dataframe.

    Returns:
        Pandas.Dataframe: Expanded chords dataframe.
    """

    # Rename columns to give them meaningful names
    df_chords_renamed = df_chords.rename(columns={
        0: 'onset_time',
        1: 'offset_time',
        2: 'key',
        3: 'degree',
        4: 'quality',
        5: 'inversion',
        6: 'roman_numeral_notation'
    })

    # We want to expand the dataset to have one row per each integer onset-time, with the chord played at that time.
    expanded_rows = []
    for i, row in df_chords_renamed.iterrows():
        if i == len(df_chords_renamed) - 1:
            break
        
        onset = int(row["onset_time"])
        offset = int(df_chords_renamed.iloc[i+1]["onset_time"])
        
        # Repeat the chord for every integer timestamp in the range [onset, offset)
        for t in range(onset, offset):
            # Handle secondary chords
            if type(row["degree"]) == str and '/' in row["degree"]:
                d1, d2 = row["degree"].split('/')[0], row["degree"].split('/')[1]
                if '+' in d2:
                    d2 = d2[1]
                    degree = (int(d1) + int(d2)) if (int(d1) + int(d2)) <= 7 else (int(d1) + int(d2) - 7)
                    degree = '+' + str(degree)
                elif '-' in d2:
                    d2 = d2[1]
                    degree = (int(d1) - int(d2)) if (int(d1) - int(d2)) >= 1 else (int(d1) - int(d2) + 7)
                    degree = '-' + str(degree)
                else:
                    degree = (int(d1) + int(d2)) if (int(d1) + int(d2)) <= 7 else (int(d1) + int(d2) - 7)
                    degree = str(degree)
            else:
                degree = row["degree"]

            # Convert string values to integers and append to the expanded dataset
            expanded_rows.append({
                "time": t,
                "key": chord_key_to_int[row["key"]],
                "degree": scale_degree_to_int[str(degree)],
                "quality": chord_quality_to_int[row["quality"]],
                "inversion": int(row["inversion"]),
                "roman_numeral_notation": row["roman_numeral_notation"]
            })
    
    expanded_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    return expanded_df

def prepare_dataset(dataset_path, train_test_split=0.8):
    """
    Split the dataset into datapoints by their phrase number (each phrase is one datapoint).
    Each datapoint is a Python dictionary containing source and target values as Pytorch tensors.

    Args:
        dataset_path (str): Path to the (previously processed) dataset.

    Returns:
        data_train, data_test (list, list): Dataset split into train and test.
    """

    data_train = []
    data_test = []
    num_dirs = len(os.listdir(dataset_path))

    for dir in os.listdir(dataset_path):
        # Load the data
        notes_df = pd.read_csv(os.path.join(dataset_path, dir, 'notes.csv'))
        chords_df = pd.read_csv(os.path.join(dataset_path, dir, 'chords.csv'))
        phrases_df = pd.read_excel(os.path.join(dataset_path, dir, 'phrases.xlsx'), header=None)

        # Process each phrase
        for _, phrase in phrases_df.iterrows():
            phrase_begin = phrase[0]
            phrase_end = phrase[1]

            # Extract notes and chords within the phrase
            notes_in_phrase = notes_df[
                (notes_df['onset_time'] >= phrase_begin) & 
                (notes_df['onset_time'] <= phrase_end)
            ]
            chords_in_phrase = chords_df[
                (chords_df['time'] >= phrase_begin) & 
                (chords_df['time'] <= phrase_end)
            ]

            # Skip empty phrases if there are no notes or chords
            if notes_in_phrase.empty or chords_in_phrase.empty:
                continue

            # Convert to tensors
            src_notes = torch.tensor(notes_in_phrase['midi_note'].values, dtype=torch.long)
            src_onsets = torch.tensor(notes_in_phrase['onset_time'].values, dtype=torch.float32) - phrase_begin

            tgt_chords = torch.tensor(chords_in_phrase[['key', 'degree', 'quality', 'inversion']].values, dtype=torch.long)
            tgt_onsets = torch.tensor(chords_in_phrase['time'].values, dtype=torch.float32) - phrase_begin

            # Append to the list
            if int(dir) <= num_dirs*train_test_split:
                data_train.append({
                    'src_notes': src_notes,
                    'src_onsets': src_onsets,
                    'tgt_chords': tgt_chords,
                    'tgt_onsets': tgt_onsets
                })
            else:
                data_test.append({
                    'src_notes': src_notes,
                    'src_onsets': src_onsets,
                    'tgt_chords': tgt_chords,
                    'tgt_onsets': tgt_onsets
                })
    
    return data_train, data_test


def process_dataset(dataset_path):
    """
    Process the dataset using process_notes_df() and process_chords_df() functions.
    Saves the result of the processing in an output directory.

    Args:
        dataset_path (str): Path to the unprocessed dataset.
    """

    for dir in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, dir)):
            continue
        
        newpath = 'data/processed/' + dir
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        df_notes = pd.read_csv(os.path.join(dataset_path, dir, 'notes.csv'), header=None)
        df_notes_clean = process_notes_df(df_notes)
        df_notes_clean.to_csv(os.path.join('data/processed/', dir, 'notes.csv'), index=False)

        df_chords = pd.read_excel(os.path.join(dataset_path, dir, 'chords.xlsx'), header=None)
        df_chords_clean = process_chords_df(df_chords)
        df_chords_clean.to_csv(os.path.join('data/processed/', dir, 'chords.csv'), index=False)

        shutil.copyfile(os.path.join(dataset_path, dir, 'phrases.xlsx'), os.path.join('data/processed/', dir, 'phrases.xlsx'))


def main():
    print("Processing dataset started...")
    process_dataset('data/functional-harmony/BPS_FH_Dataset')
    print("Processing dataset completed.")

    data_train, data_test = prepare_dataset('data/processed')
    print("Number of phrases in the train set:", len(data_train))
    print("Number of phrases in the test set:", len(data_test))


if __name__ == '__main__':
    main()
