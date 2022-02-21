import torch.nn as nn
import torch


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
        # Feel free to rename what you like, this is just a rough outline to work around
        embedded = self.embedder.embed(input)

        embedded = self.dropout(embedded)
        outputs, _= self.model(embedded)
        return self.squisher(self.fully_connected(outputs))

 
class Example():
    def __init__(self, skip):
        self.embbeddings = {
            "the" : torch.tensor([1.5, 0.2, -1, -0.3]),
            "cat" : torch.tensor([2.3, 0.1, 0, -5]),
            "sat" : torch.tensor([0.5, 0.3, -2, -1.3]),
            "dog" : torch.tensor([-0.2, 0.8, 0.3, -2]),  
            "ran" : torch.tensor([-1.4, 0.7, -1, 1.1]),
            "PAD" : torch.tensor([0, 0, 0, 0])
        }

    def embed(self, sentences):
        if type(sentences[0]) is str:
            sentences = [sentences]

        embedded = []

        for sentence in sentences:
            sent_embedded = []
            for word in sentence:
                sent_embedded.append(self.embbeddings[word])

            sent_embedded = torch.stack(sent_embedded)
            embedded.append(sent_embedded)

        embedded = torch.stack(embedded)
        return embedded

def run_example():
    # Example sentences, consider a single batch during training
    example_sentences = [["the", "cat", "sat", "PAD"],["the", "dog", "ran", "PAD"], ["cat", "dog", "cat", "dog"]]

    example_network = BiasDetector(4, 5, Example, model="LSTM", bidirectional=True)

    return example_network(example_sentences)