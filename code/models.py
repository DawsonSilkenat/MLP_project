import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizerFast, BertModel
import numpy as np


class BiasDetector(nn.Module):
    def __init__(self, hidden_dim, embedder, lr=0.0001, model="rnn", num_layers=1, dropout=0, bidirectional=False): 
        super(BiasDetector, self).__init__()
        
        if torch.cuda.device_count() > 0:
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')  
        self.to(self.device)

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

        if num_layers == 0:
            hidden_dim = embedding_dim
            self.model = torch.nn.Identity
        else:
            self.model = model(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        # While it may be sensable to use one output and interprate as a probability, easier just to have two classes
        self.fully_connected = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 2)

        # There are additional parameters we can specify if we want to do fine tuning, but probably not the time available
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, input):
        embedded = self.embedder(input)
        embedded = self.dropout(embedded)
        outputs, _ = self.model(embedded)
        return self.fully_connected(outputs)

    # This function primarly exists for testing purposes
    def get_embedder(self):
        return self.embedder

    def tokenize(self, seq):
        return self.embedder.tokenize(seq)

    def run_train_iter(self, values, targets):
        self.train() 
        
        # Updates values and targets to be approprate tensors 
        targets = self.embedder.update_targets(values, targets)
        values = self.tokenize(values)
        values, targets = values.to(device=self.device), targets.to(device=self.device)

        # Get network predictions from forward
        output = self.forward(values)

        # Compute the backwards step of backprop
        loss = F.cross_entropy(input=output.permute(0,2,1), target=targets, ignore_index=-1) 
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()

        _, indices = torch.max(output.data, 2)

        matrix = self.confusion_matrix(indices.tolist(), targets.tolist())

        return loss.data.numpy(), matrix


    def evaluate_on(self, eval_provider):
        self.eval() 
        
        matrix = np.zeros((2,2), dtype=np.uint)
        loss = []
        for targets, values in eval_provider:
            # Updates values and targets to be approprate tensors 
            targets = self.embedder.update_targets(values, targets)
            values = self.tokenize(values)
            values, targets = values.to(device=self.device), targets.to(device=self.device)

            # Get network predictions from forward
            output = self.forward(values)

            # Compute the backwards step of backprop
            loss.append(F.cross_entropy(input=output.permute(0,2,1), target=targets, ignore_index=-1)) 

            _, indices = torch.max(output.data, 2)

            matrix += self.confusion_matrix(indices.tolist(), targets.tolist())

        loss = np.mean(loss)
        return loss.data.numpy(), matrix

    def run_experiments(self, train_data, eval_data, num_epochs=100):
        summary = {"train_confusion": [], "train_loss": [], "val_confusion": [], "val_loss": []} 

        for i in range(num_epochs):
            train = iter(train_data)
            eval = iter(eval_data)

            epoch_train_confusion = np.zeros((2,2), dtype=np.uint)
            epoch_train_loss = []
            for values, targets in train: 
                loss, matrix = self.run_train_iter(values, targets)
                epoch_train_confusion += matrix
                epoch_train_loss.append(loss)

            epoch_train_loss = np.mean(epoch_train_loss)

            summary["train_confusion"].append(epoch_train_confusion)
            summary["train_loss"].append(epoch_train_loss)

            epoch_val_loss, epoch_val_confusion = self.evaluate_on(eval)
            summary["val_confusion"].append(epoch_val_confusion)
            summary["val_loss"].append(epoch_val_loss)

            print("Epoch i\ntraining loss: {:.4f}, training confusion: {}\neval loss: {:.4f}, eval confusion: {}\n"
                .format(epoch_train_loss,epoch_train_confusion,epoch_val_loss,epoch_val_confusion))


    def confusion_matrix(self, predictions, actual):
        matrix = np.zeros((2,2), dtype=np.uint)
        for pred, act in zip(predictions, actual):
            if actual != -1:
                matrix[pred, act] += 1
        return matrix

        


class BertEmbedding(nn.Module):
    def __init__(self):
        super(BertEmbedding, self).__init__()
        bert_version = "bert-base-uncased"
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_version)
        self.embedder = BertModel.from_pretrained(bert_version)

    def get_embedding_dim(self):
        return self.embedder.config.to_dict()["hidden_size"]

    def tokenize(self, seq):
        # We expect a list of list of strings, which we need to turn back into a list on sentences
        if type(seq[0]) is list:
            temp = []
            for s in seq: 
                temp.append(" ".join(s))
            seq = temp
        
        # Note this includes an attention mask
        return self.tokenizer(seq, padding=True, return_tensors='pt', return_token_type_ids=False)

    # Embedding given a sequence of words
    def embed_words(self, seq):
        return self.embed_tokens(self.tokenize(seq))

    # Embedding given a tensor of tokens, such as the one returned by tokenize
    def embed_tokens(self, tokens):
        embeddings = self.embedder(tokens["input_ids"], tokens["attention_mask"]).last_hidden_state
        return embeddings 
    
    # For simplicity assume the input has already been tokenized
    def forward(self, input):
        return self.embed_tokens(input)

    """The bert tokenizer can split one word into multiple tokens. We would like to make a single classification per word,
    however we end up with one classification per token. To adjust for this we choose to asign the same target to each token that 
    makes up a word"""
    def update_targets(self, seq, targets):
        new_targets = []
        max_length = -1
        for i in range(len(seq)):
            seq_target = []

            for j in range(len(seq[i])):
                word = seq[i][j]
                tokenized = self.tokenizer(word, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)["input_ids"]
                seq_target = seq_target + [targets[i][j] for _ in range(len(tokenized))]

            max_length = max(max_length, len(seq_target))
            seq_target = torch.tensor(seq_target, dtype=int)
            new_targets.append(seq_target)

        # Pad each tensor individually so they have the same dimension. Use -1 for padding so it can be ignored later,
        # add and extra pad on both ends to account for the start and end tokens
        new_targets = [F.pad(seq_target, (1, max_length - len(seq_target) + 1), value=-1) for seq_target in new_targets]

        # Stack to be a single tensor
        new_targets = torch.stack(new_targets)

        return new_targets

