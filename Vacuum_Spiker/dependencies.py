import torch
import pandas as pd
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import PostPre
from datetime import datetime

# Functions for processing with Spiking Neural Networks (SNNs).

def reset_voltages(network, device):
    """
    Reset the voltages of neurons in layer 'B' to -65.
    """
    network.layers['B'].v = torch.full(network.layers['B'].v.shape, -65).to(device)
    return network

def pad_to_multiple(data, T):
    """
    Pad the data with NaN rows so its length is a multiple of T.
    """
    length = len(data)
    padded_length = ((length // T) + 1) * T
    additional_length = padded_length - length
    if additional_length > 0:
        nan_rows = pd.DataFrame(np.nan, index=range(additional_length), columns=data.columns)
        data = pd.concat([data, nan_rows], ignore_index=True)
    return data

def encode_spike(x, q1, q2):
    """
    Return 1 (spike) if x is in [q1, q2), else 0. Used for data encoding.
    """
    s = torch.zeros_like(x)
    s[(x >= q1) & (x < q2)] = 1
    return s

def prepare_data(data, T, intervals, R, device, is_train=False):
    """
    Prepare and encode time series data for SNN input.
    Returns a list of sequences split by exposure time T.
    """
    series = torch.FloatTensor(data['value'])
    # Clamp values to quantile range
    series[series < torch.min(intervals)] = torch.min(intervals)
    series[series > torch.max(intervals)] = torch.max(intervals)
    # Encode data into spikes
    series2input = torch.cat([series.unsqueeze(0)] * R, dim=0)
    for i in range(R):
        series2input[i, :] = encode_spike(series2input[i, :], intervals[i], intervals[i + 1])
    # Split into sequences of length T
    sequences = torch.split(series2input.to(device), T, dim=1)
    if is_train:
        # Remove last sequence during training
        sequences = sequences[0:len(sequences) - 1]
    return sequences

def create_network(R, T, n, threshold, decay, nu1, nu2, recurrent, device):
    """
    Create and configure a spiking neural network with input and LIF layers.
    Optionally adds a recurrent connection.
    Returns the network and spike/voltage monitors.
    """
    network = Network()
    # Input and internal layers
    source_layer = Input(n=R, traces=True)
    target_layer = LIFNodes(n=n, traces=True, thresh=threshold, tc_decay=decay)
    network.add_layer(layer=source_layer, name="A")
    network.add_layer(layer=target_layer, name="B")
    # Forward connection
    forward_connection = Connection(
        source=source_layer,
        target=target_layer,
        w=0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n),
        update_rule=PostPre, nu=nu1
    )
    network.add_connection(connection=forward_connection, source="A", target="B")
    if recurrent:
        # Recurrent connection with slightly negative weights
        recurrent_connection = Connection(
            source=target_layer,
            target=target_layer,
            w=0.025 * (torch.eye(target_layer.n) - 1),
            update_rule=PostPre, nu=nu2
        )
        network.add_connection(connection=recurrent_connection, source="B", target="B")
    network = network.to(device)
    # Monitors for spikes and voltages
    source_monitor = Monitor(
        obj=source_layer,
        state_vars=("s",),
        time=T,
        device=device,
    )
    target_monitor = Monitor(
        obj=target_layer,
        state_vars=("s", "v"),
        time=T,
        device=device,
    )
    network.add_monitor(monitor=source_monitor, name="X")
    network.add_monitor(monitor=target_monitor, name="Y")
    return [network, source_monitor, target_monitor]

def run_network(sequences, network, source_monitor, target_monitor, T):
    """
    Run the SNN on the provided sequences for training or evaluation.
    Returns spike counts and the network.
    """
    sp0 = []  # Input spikes
    sp1 = []  # Output spikes
    for idx, seq in enumerate(sequences, 1):
        print(f'Running sequence {idx}')
        inputs = {'A': seq.T}
        start = datetime.now()
        network.run(inputs=inputs, time=T)
        end = datetime.now()
        print(end - start)
        spikes = {
            "X": source_monitor.get("s"),
            "B": target_monitor.get("s")
        }
        sp0.append(spikes['X'].sum(axis=2))
        sp1.append(spikes['B'].sum(axis=2))
        voltages = {"Y": target_monitor.get("v")}
    sp0 = torch.concatenate(sp0)
    sp0 = sp0.detach().cpu().numpy()
    sp1 = torch.concatenate(sp1)
    sp1 = sp1.detach().cpu().numpy()
    return [sp0, sp1, network]