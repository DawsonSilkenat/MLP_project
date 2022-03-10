import torch.nn as nn


class BiasDetector(nn.Module):
    def __init__(self, hidden_dim, embedder, model="rnn", num_layers=1, dropout=0, bidirectional=False): #, squisher=nn.Sigmoid):
        super(BiasDetector, self).__init__()

        self.embedder = embedder() 
        self.dropout = nn.Dropout(dropout)

        model = model.lower()
        if model == "rnn":
            model = nn.RNN 
        elif model == "lstm":
            model = nn.LSTM
        else:
            raise ValueError("invalid value for model argument in the BiasDetector class")

        embedding_dim = self.embedder.get_embedding_dim()
        self.model = model(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        # While it may be sensable to use one output and interprate as a probability, easier just to have two classes
        self.fully_connected = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 2)
        # self.squisher = squisher()

    def forward(self, input):
        embedded = self.embedder.embed(input)
        embedded = self.dropout(embedded)
        outputs, _ = self.model(embedded)
        return self.fully_connected(outputs)
        # return self.squisher(self.fully_connected(outputs))

    def get_embedder(self):
        return self.embedder

 
