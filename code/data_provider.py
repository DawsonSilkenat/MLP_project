from simplediff import diff
import re

# import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizerFast, BertModel

class LanguageDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_name="../bias_data/bias_data/WNC/biased.word.dev", batch_size=1, biased_label=1, unbiased_label=0, remove_stopwords=False):
        self.data = open(file_name, "r") 
        self.file_name = file_name
        self.batch_size = batch_size
        self.biased_label = biased_label
        self.unbiased_label = unbiased_label
        self.remove_stopwords = remove_stopwords

    def reset(self):
        self.data.close()
        self.data = open(self.file_name, "r")   

    def sentence_to_sequence(self, sentence):
        sequence = sentence.lower()
        # Remove non-letter characters. Note the space in the regex is deliberate 
        # TODO not editing this right now because I'm tired, but I think A-Z can be removed
        sequence = re.sub("[^A-Za-z ']+", "", sequence) 
        sequence = sequence.split()

        # Perform some cleanup of words in the sequence
        t = [] 
        for word in sequence:
            if word == "":
                continue 
            if word[-2:] == "'s":
                word = word[:-2]
            t.append(word)

        sequence = t

        return sequence

    def get_sample(self):
        for line in self.data:
            line = line.split("\t")

            # original = line[3].split()
            # updated = line[4].split()
            original = self.sentence_to_sequence(line[3])
            updated = self.sentence_to_sequence(line[4])
            
            # Simplify the data by removing the words that provide little information
            if self.remove_stopwords: 
                stop_words = stopwords.words("english")
                original = [word for word in original if word not in stop_words]
                updated = [word for word in updated if word not in stop_words]

            # Comparison between the lists
            diffs = diff(original, updated)

            # Determine the removed words
            diffs = list(filter((lambda x: x[0] != "+"), diffs))
            labels = []
            for sequence in diffs:
                if sequence[0] == "=":
                    for _ in range(len(sequence[1])):
                        labels.append(self.unbiased_label)
                elif sequence[0] == "-":
                    for _ in range(len(sequence[1])):
                        labels.append(self.biased_label)


            yield (original, labels)

    def get_batch(self):
        # Form batches of size batch_size using the data points from get_sample
        data = self.get_sample()
        values = []
        targets = []

        for point in data: 
            values.append(point[0])
            targets.append(point[1])
            # We use equality rather than inequality so that batch_size = -1 provides all points as a single batch
            if len(values) == self.batch_size:
                yield (values, targets)
                values = []
                targets = []

        # Yield the remaining elements which do not form a full batch
        if len(values) > 0:
            yield (values, targets)

    def __iter__(self):
        self.reset()
        return self.get_batch()


class BertEmbedding():
    def __init__(self):
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

    def embed(self, seq):
        tokens = self.tokenize(seq)
        embeddings = self.embedder(tokens["input_ids"], tokens["attention_mask"]).last_hidden_state
        return embeddings
    
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
        new_targets = [torch.nn.functional.pad(seq_target, (1, max_length - len(seq_target) + 1), value=-1) for seq_target in new_targets]

        # Stack to be a single tensor
        new_targets = torch.stack(new_targets)

        # TODO Need to convert this into a tensor, figure out how to indecate padding
        # We use -1 for padding so that our loss function knows to ignore these results
        return new_targets
