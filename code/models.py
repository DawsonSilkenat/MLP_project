import torch.nn as nn


class BiasDetector(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, embedder, model="rnn", num_layers=1, dropout=0, bidirectional=False, squisher=nn.Sigmoid):
        super(BiasDetector, self).__init__()

        self.embedder = embedder(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        model = model.lower()
        if model == "rnn":
            model = nn.RNN 
        elif model == "lstm":
            model = nn.LSTM
        else:
            raise ValueError("invalid value for model argument in the BiasDetector class")

        self.model = model(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fully_connected = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.squisher = squisher()

    def forward(self, input):
        embedded = self.embedder.embed(input)
        embedded = self.dropout(embedded)
        outputs, _ = self.model(embedded)
        return self.squisher(self.fully_connected(outputs))

 
