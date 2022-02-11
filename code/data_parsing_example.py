

from re import U
from simplediff import diff

if __name__ == "__main__":
    # Probably should fix the file system a bit
    data = open("bias_data/bias_data/WNC/biased.word.dev", "r")
    # Each line is a single datapoint
    line = data.readline()
    # The relevent information is split by tab characters
    line = line.split("\t")

    # These are the parts we care about, the before and after change
    original = line[3]
    updated = line[4]
    print(original)
    print(updated)
    print()

    # List of words rather than single string
    original = original.split(" ")
    updated = updated.split(" ")
    print(original)
    print(updated)
    print()

    # Comparison between the lists
    diffs = diff(original, updated)
    print(diffs)
    print()

    # The word that was changed
    diffs = list(filter((lambda x: x[0] != "="), diffs))
    print(diffs)
    print()