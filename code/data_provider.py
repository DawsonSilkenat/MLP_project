from simplediff import diff
import re 

# import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, BertModel

class LanguageDataset(torch.utils.data.IterableDataset):
    # Need to change so that 
    def __init__(self, file_name="bias_data/bias_data/WNC/biased.word.dev", sequence_length=-1, biased_label=1, unbiased_label=0, pad_label=0, require_biaswords=False, remove_stopwords=False):
        self.data = open(file_name, "r") 
        self.file_name = file_name
        self.sequence_length = sequence_length 
        self.biased_label = biased_label
        self.unbiased_label = unbiased_label
        self.pad_label = pad_label
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

            # TODO check this does what I want it to. What do I do if over the sequence length?
            if len(labels) < self.sequence_length:
                labels = labels + [self.pad_label] * (self.sequence_length - len(labels))
            
            # This is a list of words and labels. Need to convert the words to vectors
            yield (original, labels)

        # We have iterated over all elements of the file, to go over them again a call to reset should be required
        # self.data.close()
        # Well, looping and reset don't act as I would like. Fairly obvious why, 
        # the iterator just runs this function until it completes, and once empty reset doesn't change that this function has finished
        self.reset()

    def __iter__(self):
        return self.get_sample()


class BertEmbedding():
    def test(self, seq):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        sentence = " ".join(seq)

        print(sentence)
        print()
        print(tokenizer.tokenize(sentence))
        print()
        print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))


class word2vect():
    def __init__(self, data_provider):
        pass

    def to_vec(self, word):
        pass       
