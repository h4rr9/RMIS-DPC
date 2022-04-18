import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    ''' Initialize ConvGRUCell '''
    def __init__(self, input_size, hidden_size, kernel_size):

        super(ConvGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        padding = kernel_size // 2

        self.renset_gate = nn.Conv2d(input_size + hidden_size,
                                     hidden_size,
                                     kernel_size,
                                     padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size,
                                     hidden_size,
                                     kernel_size,
                                     padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size,
                                  hidden_size,
                                  kernel_size,
                                  padding=padding)

        nn.init.orthogonal_(self.resnet_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.resnet_gate.weight)
        nn.init.constant_(self.update_gate.weight)
        nn.init.constant_(self.out_gate.weight)

        def forward(self, input_tensor, hidden_state):

            if hidden_size is None:
                B, C, *spatial_dim = input_tensor.size()
                hidden_state = torch.zeros([B, self.hidden_size,
                                            *spatial_dim]).cuda()

            # [batch, channel, height, weight]

            combined = torch.cat([input_tensor, hidden_state],
                                 dim=1)  # concat along channel
            update = torch.sigmoid(self.update_gate(combined))
            reset = torch.sigmoid(self.resnet_gate(combined))
            out = torch.tanh(
                self.out_gate(
                    torch.cat([input_tensor, hidden_state * reset],
                              dim=1)))  # along channel dim
            new_state = hidden_state * (1 - update) + out * update
            return new_state


class ConvGRY(nn.Module):
    ''' Initialize a multi-layer Conv GRU '''
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size,
                 num_layers,
                 dropout=0.1):
        pass
