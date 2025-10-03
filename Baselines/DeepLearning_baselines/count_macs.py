import torch
from TSFEDLtorch.models_pytorch import YildirimOzal
from baselinas import *
from TSFEDLtorch.blocks_pytorch import RTABlock, SqueezeAndExcitationModule, DenseNetDenseBlock, DenseNetTransitionBlock
from TSFEDLtorch.utils import flip_indices_for_conv_to_lstm

# Utility functions to count MACs (Multiply-Accumulate Operations) for different deep learning models.
# This script computes the MAC counts for the different DL models used as  baselines, under the 
# different tried configurations. The results are saved in code2R.txt, and can be copied and pasted 
# in energy_measurement.R, in the Evaluation folder.


def count_lstm_macs(T, input_size, hidden_size):
    """
    Calculate MACs for an LSTM layer.
    """
    return T * (4 * hidden_size * input_size + 4 * hidden_size ** 2 + 12 * hidden_size)


def count_yildirimozal_macs(lon):
    """
    Count MACs for the YildirimOzal model.
    """
    m = YildirimOzal(input_shape=[1, lon])
    x = torch.randn((1, 1, lon))
    y = m(x)
    macs = 0
    for c in m.encoder:
        x = c(x)
        if isinstance(c, torch.nn.Conv1d):
            macs += c.in_channels * c.out_channels * c.kernel_size[0] * x.shape[-1]
        elif isinstance(c, torch.nn.BatchNorm1d):
            macs += torch.tensor(x.shape).prod()
        elif isinstance(c, torch.nn.AvgPool1d):
            macs += x.shape[-1] * x.shape[-2] * (c.kernel_size[0])
        elif isinstance(c, torch.nn.Linear):
            macs += c.in_features * c.out_features
    for c in m.decoder:
        x = c(x)
        if isinstance(c, torch.nn.Conv1d):
            macs += c.in_channels * c.out_channels * c.kernel_size[0] * x.shape[-1]
        elif isinstance(c, torch.nn.BatchNorm1d):
            macs += torch.tensor(x.shape).prod()
        elif isinstance(c, torch.nn.AvgPool1d):
            macs += x.shape[-1] * x.shape[-2] * (c.kernel_size[0])
        elif isinstance(c, torch.nn.Linear):
            macs += c.in_features * c.out_features
    result = f'macs[a$modelo=="YildirimOzal" & a$lon=={lon}]<-{macs}'
    log.write(f'{result}\n')



def count_caiwenjuan_macs(lon):
    """
    Count MACs for the AdaptiveCaiWenjuan model.
    """
    m = AdaptiveCaiWenjuan(in_features=1)
    x = torch.randn((1, 1, lon))
    y = m(x)
    macs = 0
    x1 = m.conv1(x)
    macs += m.conv1.in_channels * m.conv1.out_channels * m.conv1.kernel_size[0] * x1.shape[-1]
    x2 = m.conv2(x)
    macs += m.conv2.in_channels * m.conv2.out_channels * m.conv2.kernel_size[0] * x2.shape[-1]
    x3 = m.conv3(x)
    macs += m.conv3.in_channels * m.conv3.out_channels * m.conv3.kernel_size[0] * x3.shape[-1]
    x = torch.cat((x1, x2, x3), dim=1)
    # Dense module
    for c in m.dense_module:
        if isinstance(c, SqueezeAndExcitationModule):
            macs += torch.tensor(x.shape[0:1]).prod() * (x.shape[2] - 1)
            se = torch.mean(x, dim=2)
            se = se.view((se.size(0), 1, se.size(1)))
            for l in c.fully_connected:
                if isinstance(l, torch.nn.Linear):
                    macs += l.in_features * l.out_features * torch.tensor(se.shape[0:-1]).prod()
                se = l(se)
            se = se.view((se.size(0), se.size(2), 1))
            macs += torch.tensor(x.shape).prod()
            x = torch.multiply(x, se)
        elif isinstance(c, DenseNetTransitionBlock):
            for l in c.module:
                x = l(x)
                if isinstance(l, torch.nn.Conv1d):
                    macs += l.in_channels * l.out_channels * l.kernel_size[0] * x.shape[-1]
                elif isinstance(l, torch.nn.BatchNorm1d):
                    macs += torch.tensor(x.shape).prod()
                elif isinstance(l, torch.nn.AvgPool1d):
                    macs += x.shape[-1] * x.shape[-2] * (l.kernel_size[0])
        elif isinstance(c, DenseNetDenseBlock):
            for k in c.module:
                x1 = x.clone()
                for l in k.module:
                    x1 = l(x1)
                    if isinstance(l, torch.nn.Conv1d):
                        macs += l.in_channels * l.out_channels * l.kernel_size[0] * x1.shape[-1]
                    elif isinstance(l, torch.nn.BatchNorm1d):
                        macs += torch.tensor(x.shape).prod()
                    elif isinstance(l, torch.nn.AvgPool1d):
                        macs += x.shape[-1] * x.shape[-2] * (l.kernel_size[0])
                x = torch.cat((x1, x), dim=1)
    result = f'macs[a$modelo=="AdaptiveCaiWenjuan"]<-{macs}'
    log.write(f'{result}\n')



def count_ohshulih_macs(lon):
    """
    Count MACs for the AdaptiveOhShuLih model.
    """
    m = AdaptiveOhShuLih(in_features=1)
    x = torch.randn((1, 1, lon))
    y = m(x)
    macs = 0
    for c in m.convolutions:
        x = c(x)
        if isinstance(c, torch.nn.Conv1d):
            macs += c.in_channels * c.out_channels * c.kernel_size[0] * x.shape[-1]
        elif isinstance(c, torch.nn.BatchNorm1d):
            macs += torch.tensor(x.shape).prod()
        elif isinstance(c, torch.nn.AvgPool1d):
            macs += x.shape[-1] * x.shape[-2] * (c.kernel_size[0])
        elif isinstance(c, torch.nn.Linear):
            macs += c.in_features * c.out_features
    x = x.view(x.size(0), x.size(2), x.size(1))
    x, _ = m.lstm(x)
    macs += count_lstm_macs(x.shape[1], m.lstm.input_size, m.lstm.hidden_size)
    x = m.reduction_layer(x)
    macs += m.reduction_layer.in_channels * m.reduction_layer.out_channels * m.reduction_layer.kernel_size[0] * x.shape[-1]
    result = f'macs[a$modelo=="AdaptiveOhShuLih"]<-{macs}'
    log.write(f'{result}\n')



def count_zhengzhenyu_macs(lon):
    """
    Count MACs for the AdaptiveZhengZhenyu model.
    """
    m = AdaptiveZhengZhenyu(in_features=1)
    x = torch.randn((1, 1, lon))
    y = m(x)
    macs = 0
    for c in m.convolutions:
        x = c(x)
        if isinstance(c, torch.nn.Conv1d):
            macs += c.in_channels * c.out_channels * c.kernel_size[0] * x.shape[-1]
        elif isinstance(c, torch.nn.BatchNorm1d):
            macs += torch.tensor(x.shape).prod()
        elif isinstance(c, torch.nn.AvgPool1d):
            macs += x.shape[-1] * x.shape[-2] * (c.kernel_size[0])
        elif isinstance(c, torch.nn.Linear):
            macs += c.in_features * c.out_features
    x = flip_indices_for_conv_to_lstm(x)
    x, _ = m.lstm(x)
    macs += count_lstm_macs(x.shape[1], m.lstm.input_size, m.lstm.hidden_size)
    x = m.reduction_layer(x)
    macs += m.reduction_layer.in_channels * m.reduction_layer.out_channels * m.reduction_layer.kernel_size[0] * x.shape[-1]
    print(f'ZhengZenyu: {macs}')



def count_conv1dautoencoder_macs(lon, n_layer):
    """
    Count MACs for the Conv1dAutoencoder model.
    """
    m = Conv1dAutoencoder(torch.device('cpu'), n_layers=n_layer)
    x = torch.randn((1, 1, lon))
    y = m(x)
    macs = 0
    for c in m.encoder:
        x = c(x)
        if isinstance(c, torch.nn.Conv1d):
            macs += c.in_channels * c.out_channels * c.kernel_size[0] * x.shape[-1]
        elif isinstance(c, torch.nn.BatchNorm1d):
            macs += torch.tensor(x.shape).prod()
        elif isinstance(c, torch.nn.AvgPool1d):
            macs += x.shape[-1] * x.shape[-2] * (c.kernel_size[0])
        elif isinstance(c, torch.nn.Linear):
            macs += c.in_features * c.out_features
        elif isinstance(c, torch.nn.ConvTranspose1d):
            macs += c.in_channels * c.out_channels * c.kernel_size[0] * x.shape[-1]
    for c in m.decoder:
        x = c(x)
        if isinstance(c, torch.nn.Conv1d):
            macs += c.in_channels * c.out_channels * c.kernel_size[0] * x.shape[-1]
        elif isinstance(c, torch.nn.BatchNorm1d):
            macs += torch.tensor(x.shape).prod()
        elif isinstance(c, torch.nn.AvgPool1d):
            macs += x.shape[-1] * x.shape[-2] * (c.kernel_size[0])
        elif isinstance(c, torch.nn.Linear):
            macs += c.in_features * c.out_features
        elif isinstance(c, torch.nn.ConvTranspose1d):
            macs += c.in_channels * c.out_channels * c.kernel_size[0] * x.shape[-1]
    result = f'macs[a$modelo=="Conv1dAutoencoder" & a$lon=={lon} & a$n_lay=={n_layer}]<-{macs}'
    log.write(f'{result}\n')



def count_lstm_autoencoder_macs(lon, hidden, latent, n_layer):
    """
    Count MACs for the LSTMAutoencoder model.
    """
    m = LSTMAutoencoder(lon, hidden, latent, n_layer, torch.device('cpu'))
    x = torch.randn((1, 1, lon))
    s = x.size(1)
    macs = 0
    macs += count_lstm_macs(x.shape[1], m.encoder_lstm.input_size, m.encoder_lstm.hidden_size)
    for i in range(m.encoder_lstm.num_layers - 1):
        macs += count_lstm_macs(x.shape[1], m.encoder_lstm.hidden_size, m.encoder_lstm.hidden_size)
    _, (x, _) = m.encoder_lstm(x)
    x = m.encoder_to_latent(x[-1])
    macs += m.encoder_to_latent.in_features * m.encoder_to_latent.out_features
    x = x.unsqueeze(1).repeat(1, s, 1)
    macs += count_lstm_macs(x.shape[1], m.decoder_lstm.input_size, m.decoder_lstm.hidden_size)
    for i in range(m.decoder_lstm.num_layers - 1):
        macs += count_lstm_macs(x.shape[1], m.decoder_lstm.hidden_size, m.decoder_lstm.hidden_size)
    x, _ = m.decoder_lstm(x)
    x = m.decoder_output_layer(x)
    macs += m.decoder_output_layer.in_features * m.decoder_output_layer.out_features
    result = f'macs[a$modelo=="LSTMAutoencoder" & a$lon=={lon} & a$hidden=={hidden} & a$latent=={latent} & a$n_lay=={n_layer}]<-{macs}'
    log.write(f'{result}\n')


# Main execution: count MACs for all model configurations and log results
log = open('code2R.txt', 'w')

for lon in [50, 100, 150, 200]:
    count_yildirimozal_macs(lon)

count_caiwenjuan_macs(67)

count_ohshulih_macs(20)

for lon in [50, 100, 150, 200]:
    for n_layer in [1, 2, 3]:
        count_conv1dautoencoder_macs(lon, n_layer)

for lon in [50, 100, 150, 200]:
    for hidden in [32, 64]:
        for latent in [20, 50]:
            for n_layer in [1, 2, 3]:
                count_lstm_autoencoder_macs(lon, hidden, latent, n_layer)

log.close()
