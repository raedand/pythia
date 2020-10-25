import torch
import torch.nn as nn
from torch.autograd import Variable

# i would love to change this to an lstm but it's so fucking complicated
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.cells = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input)
        output, hidden = self.cells(input, hidden)
        output = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
