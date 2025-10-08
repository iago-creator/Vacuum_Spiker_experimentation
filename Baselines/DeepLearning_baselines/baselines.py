import torch
from TSFEDL.models_pytorch import OhShuLih, CaiWenjuan, ZhengZhenyu

"""
This module defines the deep learning baseline models for time series feature extraction.
All classes are adapted from existing architectures.
"""


class AdaptiveOhShuLih(OhShuLih):
    """
    OhShuLih model adapted for time series.
    Applies a 1D convolution to reduce channel dimension to 1.
    
    References
    ----------
        `Oh, Shu Lih, et al. "Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with
        variable length heart beats." Computers in biology and medicine 102 (2018): 278-287.`
    """
    def __init__(self, *args, **kwargs):
        super(AdaptiveOhShuLih, self).__init__(*args, **kwargs)
        self.reduction_layer = None
        self.classifier = None
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        x = super(AdaptiveOhShuLih, self).forward(x)
        if self.reduction_layer is None:
            num_channels = x.size(1)
            self.reduction_layer = torch.nn.Conv1d(in_channels=num_channels, out_channels=1, kernel_size=1).to(self.device)
        x = self.reduction_layer(x)
        return x



class AdaptiveCaiWenjuan(CaiWenjuan):
    """
    CaiWenjuan model adapted for time series.
    Adds a channel dimension for compatibility.
    
    References
    ----------
        `Cai, Wenjuan, et al. "Accurate detection of atrial fibrillation from 12-lead ECG using deep neural network."
        Computers in biology and medicine 116 (2020): 103378.`
    """
    def __init__(self, *args, **kwargs):
        super(AdaptiveCaiWenjuan, self).__init__(*args, **kwargs)
        self.classifier = None
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        x = super(AdaptiveCaiWenjuan, self).forward(x)
        x = x.unsqueeze(1)
        return x



class AdaptiveZhengZhenyu(ZhengZhenyu):
    """
    ZhengZhenyu model adapted for time series.
    Applies a 1D convolution to reduce channel dimension to 1.
    References
    ----------
        Zheng, Z., Chen, Z., Hu, F., Zhu, J., Tang, Q., & Liang, Y. (2020). An automatic diagnosis of arrhythmias using
        a combination of CNN and LSTM technology. Electronics, 9(1), 121.
    """
    def __init__(self, *args, **kwargs):
        super(AdaptiveZhengZhenyu, self).__init__(*args, **kwargs)
        self.reduction_layer = None
        self.classifier = None
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        x = super(AdaptiveZhengZhenyu, self).forward(x)
        if self.reduction_layer is None:
            num_channels = x.size(1)
            self.reduction_layer = torch.nn.Conv1d(in_channels=num_channels, out_channels=1, kernel_size=1).to(self.device)
        x = self.reduction_layer(x)
        return x



class LSTMAutoencoder(torch.nn.Module):
    """
    LSTM-based autoencoder for time series data.
    Encodes input into a latent space and reconstructs it using LSTM layers.
    References
    ---------
        Githinji, S., & Maina, C. W. (2023). Anomaly detection on time series sensor data using deep LSTM-autoencoder. 
        IEEE AFRICON 2023, Nairobi, Kenya, September 20-22, 2023, 1–6.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, device):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_dim = input_dim

        # Encoder
        self.encoder_lstm = torch.nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True).to(device)
        self.encoder_to_latent = torch.nn.Linear(hidden_dim, latent_dim).to(device)

        # Decoder
        self.latent_to_hidden = torch.nn.Linear(latent_dim, hidden_dim).to(device)
        self.decoder_lstm = torch.nn.LSTM(latent_dim, hidden_dim, n_layers, batch_first=True).to(device)
        self.decoder_output_layer = torch.nn.Linear(hidden_dim, input_dim).to(device)

    def forward(self, x):
        # Encode input sequence
        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.encoder_to_latent(hidden[-1])

        # Repeat latent vector for each time step
        decoder_input = latent.unsqueeze(1).repeat(1, x.size(1), 1)

        # Decode sequence
        output, _ = self.decoder_lstm(decoder_input)
        output = self.decoder_output_layer(output)
        return output



class Conv1dAutoencoder(torch.nn.Module):
    """
    1D Convolutional autoencoder for time series data.
    Uses configurable encoder and decoder stacks.
    References
    ---------
        Gorman, M., Ding, X., Maguire, L., & Coyle, D. (2023). Anomaly detection in batch manufacturing processes using localized reconstruction errors
        from 1-D convolutional AutoEncoders. IEEE Transactions on Semiconductor Manufacturing, 36, 147–150.
    """
    def __init__(self, device, n_layers=3, in_channels=1, base_channels=16, kernel_size=10):
        super(Conv1dAutoencoder, self).__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.kernel_size = kernel_size

        # Encoder: stack of Conv1d + ReLU layers
        encoder_layers = []
        input_channels = in_channels
        for i in range(n_layers):
            output_channels = base_channels * (2 ** i)
            encoder_layers.append(
                torch.nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size - 2) // 2).to(device)
            )
            encoder_layers.append(torch.nn.ReLU().to(device))
            input_channels = output_channels
        self.encoder = torch.nn.Sequential(*encoder_layers).to(device)

        # Decoder: stack of ConvTranspose1d + ReLU layers
        decoder_layers = []
        for i in range(n_layers - 1, 0, -1):
            output_channels = base_channels * (2 ** (i - 1))
            decoder_layers.append(
                torch.nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size - 2) // 2, output_padding=1).to(device)
            )
            decoder_layers.append(torch.nn.ReLU().to(device))
            input_channels = output_channels
        # Final layer to restore original channel count
        decoder_layers.append(
            torch.nn.ConvTranspose1d(input_channels, in_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size - 2) // 2).to(device)
        )
        decoder_layers.append(torch.nn.ReLU().to(device))
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        original_length = x.shape[-1]
        x = self.encoder(x)
        x = self.decoder(x)
        x = x[:, :, :original_length]
        return x
