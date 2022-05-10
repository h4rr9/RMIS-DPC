# from https://github.com/TengdaHan/DPC/blob/master/backbone/convrnn.py

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

        self.resnet_gate = nn.Conv2d(input_size + hidden_size,
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
        nn.init.constant_(self.resnet_gate.weight, 0)
        nn.init.constant_(self.update_gate.weight, 0)
        nn.init.constant_(self.out_gate.weight, 0)

    def forward(self, input_tensor, hidden_state):

        if hidden_state is None:
            B, C, *spatial_dim = input_tensor.size()
            hidden_state = torch.zeros([B, self.hidden_size, *spatial_dim])
            if input_tensor.is_cuda:
                hidden_state = hidden_state.cuda()

        # input := [batch, channel, height, weight]

        # concat along channel
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.resnet_gate(combined))
        out = torch.tanh(
            self.out_gate(
                torch.cat([input_tensor, hidden_state * reset],
                          dim=1)))  # along channel dim
        new_state = hidden_state * (1 - update) + out * update
        return new_state


class ConvGRU(nn.Module):
    ''' Initialize a multi-layer Conv GRU '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size,
                 num_layers,
                 dropout=0.1):
        super(ConvGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []

        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_size

            cell = ConvGRUCell(input_dim, self.hidden_size, self.kernel_size)
            name = f'ConvGRUCell_{str(i).zfill(2)}'

            setattr(self, name, cell)  # set ConGRUCell as attribute of ConvGRU
            cell_list.append(getattr(self, name))

        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, hidden_state=None):

        # grab batch and seq_len dim
        B, seq_len, *_ = x.size()

        # initialize hidden_state
        if hidden_state is None:
            hidden_state = [None] * self.num_layers

        # input := [batch, seq_len, channel, height, width]

        current_layer_input = x
        del x

        last_state_list = []

        # for each layer of ConvGRU
        for idx in range(self.num_layers):

            # grab current hidden_state
            cell_hidden = hidden_state[idx]
            output_inner = []

            for t in range(seq_len):
                # current_layer_input sliced along time dimension
                cell_hidden = self.cell_list[idx](current_layer_input[:, t, :],
                                                  cell_hidden)
                # dropout each time step
                cell_hidden = self.dropout_layer(cell_hidden)
                output_inner.append(cell_hidden)

            # stack along time (second) dim
            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output

            last_state_list.append(cell_hidden)

        # stack along time (second) dim
        last_state_list = torch.stack(last_state_list, dim=1)

        return layer_output, last_state_list


if __name__ == "__main__":
    m = ConvGRU(input_size=10, hidden_size=20, kernel_size=3, num_layers=2)
    x = torch.randn(4, 5, 10, 6, 6)  # [B, SL, C, H, W]

    if torch.cuda.is_available():
        m = m.cuda()
        x = x.cuda()

    __import__('ipdb').set_trace()
    o, h = m(x)
    print(o.shape, h.shape)
