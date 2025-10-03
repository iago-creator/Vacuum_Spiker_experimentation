import torch
import pandas as pd
import numpy as np

def split_sequences(data, min_length):
    """
    Splits the input DataFrame into subsequences of normal data (label == 0).
    Returns only subsequences longer than min_length.
    """
    intervals = []
    in_sequence = False
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
    if in_sequence:
        intervals.append((start_idx, end_idx))
    intervals_df = pd.DataFrame(intervals, columns=['start', 'end'])
    subs = []
    for _, row in intervals_df.iterrows():
        start_tmp = row['start']
        end_tmp = row['end']
        if end_tmp - start_tmp >= min_length:
            subs.append(data.iloc[start_tmp:end_tmp].reset_index(drop=True))
    return subs

def create_sequences(data, sequence_length):
    """
    Creates sliding window sequences from a 1D tensor or array.
    """
    num_sequences = len(data) - sequence_length + 1
    sequences = torch.zeros((num_sequences, sequence_length))
    for i in range(num_sequences):
        sequences[i] = data[i:i + sequence_length]
    return sequences

def batch_sequences(sequences, batch_size, include_remainder=False):
    """
    Splits sequences into batches of batch_size. Optionally includes remainder as last batch.
    """
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    batches = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batches.append(sequences[start_index:end_index])
    if include_remainder and end_index < len(sequences):
        batches.append(sequences[end_index:])
    return tuple(batches)

def prepare_data(data, seq_length, batch_size, train=False):
    """
    Prepares data for training or evaluation: creates sequences and batches, shuffles if training.
    """
    sequences = create_sequences(data, seq_length)
    if train:
        idx = torch.randperm(sequences.size(0))
        sequences = sequences[idx]
    return batch_sequences(sequences, batch_size)

def train_model(model, epoch, seqs, optimizer, device):
    """
    Trains the model for one epoch on the provided batches.
    Prints average training loss for the epoch.
    """
    loss_train = []
    for index in range(len(seqs)):
        inputs = seqs[index].unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss(outputs, inputs)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.detach().item())
    loss_train = np.array(loss_train)
    print(f"Epoch {epoch} Train loss: {np.mean(loss_train)}")
    return model

