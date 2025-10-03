
# Import necessary libraries
import torch
import pandas as pd
import numpy as np



def split_sequences(data, min_length):
    """
    Splits the training data into subsequences of normal data (label == 0).
    This is mainly used for baselines, not SNNs.
    """
    intervals = []
    in_sequence = False
    # Identify intervals of normal data
    for i in range(len(data)):
        if data.loc[i, 'label'] == 0:
            if not in_sequence:
                start_idx = i
                in_sequence = True
            end_idx = i + 1
        else:
            if in_sequence:
                intervals.append((start_idx, end_idx))
                in_sequence = False
    # Add last interval if still open
    if in_sequence:
        intervals.append((start_idx, end_idx))
    # Create DataFrame of intervals
    intervals_df = pd.DataFrame(intervals, columns=['start', 'end'])
    subs = []
    # Split data into subsequences
    for _, row in intervals_df.iterrows():
        start_tmp = row['start']
        end_tmp = row['end']
        if end_tmp - start_tmp >= min_length:
            subs.append(data.iloc[start_tmp:end_tmp].reset_index(drop=True))
    return subs




def create_sequences(data, sequence_length):
    """
    Create sequences of specified length from the input data for model training.
    """
    num_sequences = len(data) - sequence_length + 1
    sequences = np.zeros((num_sequences, sequence_length))
    for i in range(num_sequences):
        sequences[i] = data[i:i + sequence_length]
    return sequences



def prepare_data(data, seq_length, train=False):
    """
    Prepare data for training or evaluation.
    If train=True, shuffle the sequences.
    """
    sequences = create_sequences(data, seq_length)
    if train:
        # Shuffle sequences for training
        idx = np.random.permutation(sequences.shape[0])
        sequences = sequences[idx]
    # Return processed sequences
    return sequences

