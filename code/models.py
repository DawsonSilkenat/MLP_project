import torch.nn as nn
import torch


class bias_detector(nn.Module):
    def __init__(self, embedding_dim, embedder, num_layers=1, dropout=0, bidirectional=False):
        super().__init__()

        # The embedder should be a class
        # embedding_dim should be the per token dimension of the output of the embedder
        # ie, if each word is encoded individually, return 
        self.embedder = embedder(embedding_dim)

        # I believe to turn the RNN into and LSTM model, just replace RNN with LSTM in the line below. 
        # Therefor, focus on understanding vectors first
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        self.fully_connected = nn.Linear(embedding_dim, 1)

    def forward(self, input):
        # Feel free to rename what you like, this is just a rough outline to work around
        embedded = self.embedder.embed(input)

        outputs, (hidden, cell) = self.rnn(embedded)

        return nn.Sigmoid(self.fully_connected(outputs))

class test(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.LSTM(input_size=3, hidden_size=5, batch_first=False)

        self.fully_connected = nn.Linear(5, 2)
        self.endfunc = nn.Sigmoid()

    def forward(self, input):
        print(input)
        print()

        outputs, (hidden, cell) = self.rnn(input)

        print(outputs)
        print()

        return self.endfunc(self.fully_connected(outputs))

    def get_test(self):
        return torch.tensor([
            [
                [1.0,2.0,3.0],[4.0,5.0,6.0]
            ]])
    
