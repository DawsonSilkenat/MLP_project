import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizerFast, BertModel
import numpy as np
import os
import time
import csv

class BiasDetector(nn.Module):
    def __init__(self, hidden_dim, embedder, experiment_name, lr=5e-5, model="lstm", num_layers=1, dropout=0.1, bidirectional=False, class_weights=None): 
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

        # Default value for weights is already none, so we don't need to do anything if class_weights is none
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights)
            # We apply some normalisation to the weights as to not significantly increase or decrease learning rate on average
            class_weights = class_weights / class_weights.sum()

        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

        self.experiment_folder = os.path.abspath(experiment_name)

        print(self.experiment_folder)

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
        # loss = F.cross_entropy(input=output.permute(0,2,1), target=targets, ignore_index=-1) 
        loss = self.criterion(input=output.permute(0,2,1), target=targets)
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
        for values, targets in eval_provider:
            # Updates values and targets to be approprate tensors 
            targets = self.embedder.update_targets(values, targets)

            values = self.tokenize(values)
            values, targets = values.to(device=self.device), targets.to(device=self.device)

            # Get network predictions from forward
            output = self.forward(values)

            # Compute the backwards step of backprop
            # loss.append(F.cross_entropy(input=output.permute(0,2,1), target=targets, ignore_index=-1)) 
            loss.append(self.criterion(input=output.permute(0,2,1), target=targets).data)

            _, indices = torch.max(output.data, 2)

            matrix += self.confusion_matrix(indices.tolist(), targets.tolist())

            break
        

        loss = np.mean(loss)

        return loss, matrix

    def run_experiments(self, train_data, eval_data, num_epochs=100):
        # summary = {"train_confusion": [], "train_loss": [], "val_confusion": [], "val_loss": []} 

        # Rather than store matrices, summary will have a seprate list for each matrix entry
        # For instance, confusion_0_1 will be the list containing matrix[0,1] for each epoch
        summary = {"train_confusion_0_0": [], "train_confusion_0_1": [], "train_confusion_1_0": [], "train_confusion_1_1": [], "train_loss": [], 
                    "val_confusion_0_0": [], "val_confusion_0_1": [], "val_confusion_1_0": [], "val_confusion_1_1": [], "val_loss": []} 

        # There are some warnings we are ignoring, this newline helps separate the output from these warnings
        print()
        for i in range(num_epochs):
            epoch_start_time = time.time()
            train = iter(train_data)
            eval = iter(eval_data)

            epoch_train_confusion = np.zeros((2,2), dtype=np.uint)
            epoch_train_loss = []
            for values, targets in train: 
                loss, matrix = self.run_train_iter(values, targets)
                epoch_train_confusion += matrix
                epoch_train_loss.append(loss)

                break


            epoch_train_loss = np.mean(epoch_train_loss)

            summary["train_confusion_0_0"].append(epoch_train_confusion[0,0])
            summary["train_confusion_0_1"].append(epoch_train_confusion[0,1])
            summary["train_confusion_1_0"].append(epoch_train_confusion[1,0])
            summary["train_confusion_1_1"].append(epoch_train_confusion[1,1])
            summary["train_loss"].append(epoch_train_loss)

            epoch_val_loss, epoch_val_confusion = self.evaluate_on(eval)

            summary["val_confusion_0_0"].append(epoch_val_confusion[0,0])
            summary["val_confusion_0_1"].append(epoch_val_confusion[0,1])
            summary["val_confusion_1_0"].append(epoch_val_confusion[1,0])
            summary["val_confusion_1_1"].append(epoch_val_confusion[1,1])
            summary["val_loss"].append(epoch_val_loss)

            epoch_elapsed_seconds = int(time.time() - epoch_start_time) 
            estimated_remaining_seconds = epoch_elapsed_seconds * (num_epochs - i - 1)


            print(num_epochs - i - 1)
            print(estimated_remaining_seconds)
            print()

            epoch_elapsed_minutes = epoch_elapsed_seconds // 60
            epoch_elapsed_seconds = epoch_elapsed_seconds % 60
            
            print(epoch_elapsed_minutes, epoch_elapsed_seconds)
            print()

            estimated_remaining_minutes = estimated_remaining_seconds // 60
            estimated_remaining_seconds = estimated_remaining_seconds % 60
            print(estimated_remaining_minutes, estimated_remaining_seconds)
            print()

            print("Epoch {}\ntraining loss: {:.4f}, training confusion: {}\neval loss: {:.4f}, eval confusion: {}"
                .format(i, epoch_train_loss, epoch_train_confusion, epoch_val_loss, epoch_val_confusion))
            print("Epoch run time: {} minutes and {} seconds\nEstimated remaining time: {} minutes and {} seconds\n"
                .format(epoch_elapsed_minutes, epoch_elapsed_seconds, estimated_remaining_minutes, epoch_elapsed_seconds))
        
        print("saving")

        self.save_model("model.pt")
        self.save_stats(summary, "statistics.csv")

        print("done")


    def confusion_matrix(self, predictions, actual):
        matrix = np.zeros((2,2), dtype=np.uint)
        for pred, act in zip(predictions, actual):
            if actual != -1:
                matrix[pred, act] += 1
        return matrix

    def save_model(self, filename, path=None):
        if path is None:
            path = self.experiment_folder
        if not os.path.exists(path): 
            os.mkdir(path)

        file_location = os.path.join(path, filename)

        torch.save({
            "embedder" : self.embedder.embedder.state_dict(),
            "dropout" : self.dropout.state_dict(),
            "model" : self.model.state_dict(),
            "fully connected" : self.fully_connected.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "criterion" : self.criterion.state_dict() 
        }, file_location)

    def save_stats(self, stats, filename, path=None):
        if path is None:
            path = self.experiment_folder
        if not os.path.exists(path): 
            os.mkdir(path)

        file_location = os.path.join(path, filename)

        with open(file_location, "w") as f:
            writer = csv.writer(f)

            writer.writerow(list(stats.keys()))

            total_rows = len(list(stats.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats.values())]
                writer.writerow(row_to_add) 

    def load_model(self, path):
        save_data = torch.load(path)

        self.dropout.load_state_dict(save_data["dropout"])
        self.model.load_state_dict(save_data["model"])
        self.fully_connected.load_state_dict(save_data["fully connected"])
        self.optimizer.load_state_dict(save_data["optimizer"])
        self.criterion.load_state_dict(save_data["criterion"])
        self.embedder.embedder.load_state_dict(save_data["embedder"])

        


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

