import torch
from torch import nn
from torch.nn import functional as F

class TimeDistributed(nn.Module):
    """
    TimeDistributed module implementation.
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        t, n = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, *x.size()[2:]).to(torch.float32)
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, *y.size()[1:]).to(torch.float32)
        return y


def flip_indices_for_conv_to_lstm(x: torch.Tensor) -> torch.Tensor:
    """
    Changes the (N, C, L) dimension to (N, L, C). This is due to features in PyTorch's LSTMs are expected on the last dim.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    x : torch.Tensor
        Output tensor.
    """
    return x.view(x.size(0), x.size(2), x.size(1))

def flip_indices_for_conv_to_lstm_reshape(x: torch.Tensor) -> torch.Tensor:
    """
    Changes the (N, C, L) dimension to (N, L, C). This is due to features in PyTorch's LSTMs are expected on the last dim.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    x : torch.Tensor
        Output tensor.
    """
    return x.reshape(x.size(0), x.size(2), x.size(1))

