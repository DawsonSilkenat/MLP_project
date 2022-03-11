from simplediff import diff
import re

# import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
import torch


class LanguageDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_name, batch_size=1, biased_label=1, unbiased_label=0, remove_stopwords=False):
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


