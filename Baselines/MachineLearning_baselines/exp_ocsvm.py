
# Import required libraries
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from dependencies import *

# Main script for final experimentation with One-Class SVM
parser = argparse.ArgumentParser(description='Experimentation with One-Class SVM.')
parser.add_argument('-l', '--length', type=int, help='Sequence length.')
parser.add_argument('-n', '--nu', type=float, help='Learning parameter (nu).')
parser.add_argument('-p', '--path', type=str, help='Path to data file.')
args = parser.parse_args()
lon = args.length
nu = args.nu
path = args.path
model = 'ocsvm'

# Read input data (must have 'label' and 'value' columns)
data = pd.read_csv(path, na_values=['NA'])
# Ensure correct types
data['value'] = data['value'].astype('float64')
data['label'] = data['label'].astype('Int64')
# Set missing labels to 0
data.loc[data['label'].isna(), 'label'] = 0

# Cross-validation setup
tscv = TimeSeriesSplit(n_splits=5)
split = 0
for train_index, test_index in tscv.split(data):
    split += 1
    # Split data into train and test sets
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # Scale values using StandardScaler
    scaler = StandardScaler()
    # Mark missing values in train as anomalies
    data_train.loc[data_train['value'].isna(), 'label'] = 1
    scaler.fit(data_train[['value']][data_train['label'] != 1])
    data_train['value'] = scaler.transform(data_train[['value']])
    data_test['value'] = scaler.transform(data_test[['value']])

    # Split training data into windows of normal sequences
    data_train = split_sequences(data_train, lon)

    if not len(data_train):
        print('Window size too large.')
        sys.exit()

    # Prepare training data for One-Class SVM
    data_train2 = []
    for d in data_train:
        data_train2.append(prepare_data(np.array(d['value'], dtype=np.float32).reshape(d.shape[0]), lon, train=True))
    if not len(data_train2):
        sys.exit()
    data_train2 = np.concatenate(data_train2)

    # Remove NaN sequences from training data
    row_sums = data_train2.sum(axis=1)
    inds = np.arange(len(row_sums))[np.isnan(row_sums)]
    data_train2 = np.delete(data_train2, inds, axis=0)

    # Prepare test sequences and remove NaNs
    seqs_test = create_sequences(np.array(data_test['value'], dtype=np.float32).reshape(data_test.shape[0]), lon)
    row_sums = seqs_test.sum(axis=1)
    inds = np.arange(len(row_sums))[np.isnan(row_sums)]
    seqs_test = np.delete(seqs_test, inds, axis=0)
    label = data_test['label'][lon - 1:].values
    label = np.delete(label, inds)

    # Train One-Class SVM model
    m = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')

    # Prepare output directories and file paths
    output_path = f'/output/MLBaselines/{model}/{path}/{lon}/{nu}'
    os.makedirs(output_path, exist_ok=True)
    path_traza = f'{output_path}/traza_{split}'
    path_label = f'{output_path}/label_{split}'
    path_value = f'{output_path}/value_{split}'
    path_vectors = f'{output_path}/n_vectors_{split}'

    # Fit model and predict anomalies
    m.fit(data_train2)
    n_support = m.n_support_
    preds = m.predict(seqs_test)
    preds = ((-1) * preds + 1) / 2

    # Save results
    np.savetxt(path_traza, preds, delimiter=',')
    np.savetxt(path_label, label, delimiter=',')
    np.savetxt(path_value, np.array(data_test['value'][lon - 1:]), delimiter=',')
    np.savetxt(path_vectors, n_support, delimiter=',')
    

