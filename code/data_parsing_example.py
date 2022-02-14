from typing import Sequence
from simplediff import diff
import re 

# import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # Consider using this to trim words, just not sure how it will work with stopwords
import torch



class Language_dataset(torch.utils.data.IterableDataset):
    def __init__(self, file_name="bias_data/bias_data/WNC/biased.word.dev", stem=False, remove_stopwords=False):
        self.data = open(file_name, "r") 
        self.file_name = file_name 
        self.stem = stem
        self.remove_stopwords = remove_stopwords

    def reset(self):
        self.data = open(self.file_name, "r")   

    def get_sample(self):
        for line in self.data:
            # # Each line is a single datapoint; The relevent information is split by tab characters
            # line = self.data.readline()
            line = line.split("\t")

            # These are the parts we care about, the before and after change
            original = line[3]
            updated = line[4]

            original = original.lower()
            updated = updated.lower()

            # Where do I apply this? Needs 
            # if self.stem:
            #     ps = PorterStemmer()
            # else:
            #     ps = lambda x: x

            # Remove non-letter characters. Note the space in the regex is deliberate 
            original = re.sub("[^A-Za-z ']+", "", original) 
            updated = re.sub("[^A-Za-z ']+", "", updated) 

            # List of words rather than single string
            original = original.split()
            updated = updated.split()
            
            # Simplify the data by removing the words that provide little information
            if self.remove_stopwords: 
                stop_words = stopwords.words("english")
                original = [word for word in original if word not in stop_words and word != ""]
                updated = [word for word in updated if word not in stop_words and word != ""]

            print(original)
            print(updated)

            # Comparison between the lists
            diffs = diff(original, updated)

            print(diffs)

            # Find the index of every removed word
            diffs = list(filter((lambda x: x[0] != "+"), diffs))
            indeces = []
            index = 0
            for sequence in diffs:
                if sequence[0] == "=":
                    index += len(sequence[1]) 
                elif sequence[0] == "-":
                    for _ in range(len(sequence[1])):
                        indeces.append(index)
                        index += 1
                else:
                    print("Unexpected behavour when determining indices") 
                
            print(line[3])
            print()
            print(line[4])
            print()
            for i in indeces:
                print(original[i])
                print()
            yield (original, indeces)

        # We have iterated over all elements of the file, to go over them again a call to reset should be required
        self.data.close()

        # # We specifically care about the words that were removed 
        # diffs = list(filter((lambda x: x[0] == "-"), diffs))
        # diffs = [word for diff in diffs for word in diff[1:]]

    def __iter__(self):
        return self.get_sample()


        

