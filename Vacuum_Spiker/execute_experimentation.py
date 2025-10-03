import torch
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import TimeSeriesSplit
from dependencies import *

# Main script for SNN experimentation. Configurable parameters are provided via command line arguments.
parser = argparse.ArgumentParser(description='Experimentation with STDP.')


parser.add_argument('-n1', '--nu1', type=str, help='nu parameter for the first layer.')
parser.add_argument('-n2', '--nu2', type=str, help='nu parameter for the second layer.')
parser.add_argument('-n', '--neurons', type=int, help='Number of neurons in the second layer.')
parser.add_argument('-th', '--threshold', type=float, help='Neuron firing threshold.')
parser.add_argument('-d', '--decay', type=float, help='Voltage decay constant.')
parser.add_argument('-a', '--extension', type=float, help='Percentage of range extension for input data.')
parser.add_argument('-r', '--resolution', type=float, help='Input data resolution.')
parser.add_argument('-p', '--path', type=str, help='Data path.')
parser.add_argument('-e', '--epochs', type=str, help='Epochs.')
parser.add_argument('-rc', '--recurrent', type=str, help='With or without recurrent layer.')

split=0

# Parse command line arguments
args = parser.parse_args()
nu1 = eval(args.nu1)
nu2 = eval(args.nu2)
n = args.neurons
threshold = args.threshold
decay = args.decay
exten = args.extension
path = args.path
epochs = eval(args.epochs)
reso = args.resolution
recurrent = eval(args.recurrent)

# Set device to CUDA (GPU)
device = torch.device('cuda')

# Prepare output directory name strings
nu1_str = str(nu1).replace('(', '').replace(',', '_').replace(')', '')
nu2_str = str(nu2).replace('(', '').replace(',', '_').replace(')', '')

# Sequence length for training/testing
T = 100

# Read input data. Must contain 'label' and 'value' columns.
data = pd.read_csv(path, na_values=['NA'])

# Ensure correct data types
data['value'] = data['value'].astype('float64')
data['label'] = data['label'].astype('Int64')

# Set missing label values to 0 for consistency
data.loc[data['label'].isna(), 'label'] = 0

# Prepare cross-validation (TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=5)
split = 0

for train_index, test_index in tscv.split(data):
    split += 1
    # Split data into train and test sets for this fold
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    data_train['value'] = data_train['value'].astype('float64')
    data_test['label'] = data_test['label'].astype('Int64')

    # Get min and max values from non-anomalous training data
    minim = min(data_train['value'][data_train['label'] != 1])
    maxim = max(data_train['value'][data_train['label'] != 1])

    # Set value to NaN for anomalous points in training set
    data_train.loc[data_train['label'] == 1, 'value'] = np.nan

    # Model input ranges
    data_range = maxim - minim

    # Build intervals for input encoding
    one = torch.FloatTensor(np.arange(minim - exten * data_range, minim, reso * data_range))
    two = torch.FloatTensor(np.arange(maxim, maxim + exten * data_range, reso * data_range))
    half = torch.unique(torch.quantile(torch.FloatTensor(data_train['value'].dropna()), torch.FloatTensor(np.arange(0, 1, reso))))
    intervals = torch.cat((one, half, two))

    # Number of input neurons
    R = len(intervals) - 1

    # Create SNN
    network, source_monitor, target_monitor = create_network(R, T, n, threshold, decay, nu1, nu2, recurrent, device)

    # Pad test set to multiple of T
    data_test = pad_to_multiple(data_test, T)

    # Prepare training and test sequences
    network.learning = True
    secuencias2train = prepare_data(data_train, T, intervals, R, device, is_train=True)
    print(f'Training dataset length: {len(secuencias2train)}')
    secuencias2test = prepare_data(data_test, T, intervals, R, device)
    print(f'Test dataset length: {len(secuencias2test)}')

    # Training loop
    for e in range(max(epochs)):
        network.learning = True
        print(f'Epoch {e}')
        network = reset_voltages(network, device)
        spikes_input, spikes, network = run_network(secuencias2train, network, source_monitor, target_monitor, T)

        if e + 1 in epochs:
            # Validation step
            network.learning = False
            network = reset_voltages(network, device)
            spikes_input, spikes, network = run_network(secuencias2test, network, source_monitor, target_monitor, T)

            # Save results to output files
            output_path = f'/output/Vacuum_Spiker/{path}/{nu1_str}/{nu2_str}/{n}/{threshold}/{decay}/{exten}/{e+1}/{reso}/{recurrent}'
            os.makedirs(output_path, exist_ok=True)
            np.savetxt(f'{output_path}/spikes_{split}', spikes, delimiter=',')
            np.savetxt(f'{output_path}/label_{split}', np.array(data_test['label']), delimiter=',')
            np.savetxt(f'{output_path}/value_{split}', np.array(data_test['value']), delimiter=',')

