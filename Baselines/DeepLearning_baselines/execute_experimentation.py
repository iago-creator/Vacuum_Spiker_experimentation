import torch, pandas as pd, numpy as np, os, argparse, sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from dependencies import *
from baselines import *
from TSFEDL.models_pytorch import YildirimOzal

# Main script for final experimentation.
# Configurable parameters are provided via command line arguments.
parser = argparse.ArgumentParser(description='Deep Learning baselines experimentation.')


parser.add_argument('-l', '--length', type=int, help='Sequence length.')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch size.')
parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate.')
parser.add_argument('-e', '--epochs', type=str, help='Number of epochs.')
parser.add_argument('-p', '--path', type=str, help='Data path.')
parser.add_argument('-m', '--model', type=str, help='Model to use.')
parser.add_argument('-hi', '--hidden', type=int, help='Hidden layer dimension.')
parser.add_argument('-lat', '--lat', type=int, help='Latent space dimension.')
parser.add_argument('-n', '--n', type=int, help='Number of encoder (decoder) layers.')


# Parse command line arguments
args = parser.parse_args()
lon = args.length
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = eval(args.epochs)
path = args.path
model = args.model
hidden = args.hidden
latent = args.lat
n_layer = args.n

# Set device to CUDA (GPU)
device = torch.device('cuda')

# Adjust sequence length depending on model type
if model == 'AdaptiveOhShuLih':
    lon = 20
elif model == 'AdaptiveCaiWenjuan':
    lon = 67
elif model == 'AdaptiveZhengZhenyu':
    lon = 256

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

print('Entering main loop')

for train_index, test_index in tscv.split(data):
    split += 1
    # Split data into train and test sets for this fold
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    data_train = data_train.reset_index()
    data_test = data_test.reset_index()

    # Scale data using StandardScaler (fit only on non-anomalous values)
    scaler = StandardScaler()
    # Mark missing values in train as anomalies (label=1)
    data_train.loc[data_train['value'].isna(), 'label'] = 1
    scaler.fit(data_train[['value']][data_train['label'] != 1])
    data_train['value'] = scaler.transform(data_train[['value']])
    data_test['value'] = scaler.transform(data_test[['value']])

    # Split train data into sequences by label
    data_train = split_sequences(data_train, lon)
    if not len(data_train):
        print('Window size too large.')
        sys.exit()  # No training possible

    # Convert train data to tensors and create batches
    data_train2 = []
    for d in data_train:
        data_train2.extend(prepare_data(torch.FloatTensor(d['value']).reshape(d.shape[0]), lon, batch_size, train=True))

    if not len(data_train2):
        sys.exit()

    data_train2 = tuple(data_train2)

    # Prepare test sequences
    seqs_test = create_sequences(torch.FloatTensor(data_test['value']).reshape(data_test.shape[0]), lon).unsqueeze(1).to(device)

    # Model selection and initialization
    if model == 'YildirimOzal':
        m = YildirimOzal(input_shape=[1, lon]).to(device)
    elif model == 'LSTMAutoencoder':
        m = LSTMAutoencoder(lon, hidden, latent, n_layer, device)
    elif model == 'Conv1dAutoencoder':
        m = Conv1dAutoencoder(device, n_layers=n_layer)
    else:
        m = eval(model)(in_features=1).to(device)

    # Create optimizer
    if not model == 'LSTMAutoencoder' and not model == 'Conv1dAutoencoder':
        optimizer = m.optimizer_(m.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    # Set loss function (MSE)
    m.loss = torch.nn.MSELoss()

    # Training loop
    for epoch in range(max(epochs)):
        output_path = f'output/{model}/{path}/{lon}/{batch_size}/{learning_rate}/{epoch+1}/{hidden}/{latent}/{n_layer}'
        os.makedirs(output_path, exist_ok=True)
        # Skip training if output files already exist
        path_traza = f'{output_path}/traza_{split}'
        path_label = f'{output_path}/label_{split}'
        path_value = f'{output_path}/value_{split}'

        if os.path.exists(path_traza) and os.path.exists(path_label) and os.path.exists(path_value):
            continue

        m.train()
        m = train_model(m, epoch, data_train2, optimizer, device)
        if epoch+1 in epochs:
            # Validation step
            m.eval()
            with torch.no_grad():
                preds = m(seqs_test)

            # For LSTMAutoencoder, use MAE and evaluate per point
            if model == 'LSTMAutoencoder':
                error = torch.abs(preds - seqs_test).squeeze(1)
                print(f'error shape: {error.shape}')
                p = torch.FloatTensor([]).to(device)
                for k in range(1, error.shape[0] - error.shape[1] + 1):
                    r = 0
                    for i in range(error.shape[1]):
                        r += error[k + i, error.shape[1] - 1 - i]
                    r = r / error.shape[1]
                    p = torch.cat((p, r.unsqueeze(0)), dim=0)
                p = p.detach().cpu().numpy()
            else:
                p = torch.mean(torch.pow(preds - seqs_test, 2), axis=2).detach().cpu().numpy()

            # Save results to output files
            output_path = f'/output/DLBaselines/{model}/{path}/{lon}/{batch_size}/{learning_rate}/{epoch+1}/{hidden}/{latent}/{n_layer}'
            os.makedirs(output_path, exist_ok=True)

            np.savetxt(path_traza, p, delimiter=',')
            np.savetxt(path_label, np.array(data_test['label'][lon-1:]), delimiter=',')
            np.savetxt(path_value, np.array(data_test['value'][lon-1:]), delimiter=',')


