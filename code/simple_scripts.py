import data_provider
import models 

model = models.BiasDetector(1, models.BertEmbedding, "test", num_layers=0)
# train = data_provider.LanguageDataset("../bias_data/bias_data/WNC/biased.word.train", batch_size=1, remove_stopwords=True)
# eval = data_provider.LanguageDataset("../bias_data/bias_data/WNC/biased.word.test", batch_size=1, remove_stopwords=True)


train = data_provider.LanguageDataset("bias_data_reduced/biased.word.train", batch_size=1, remove_stopwords=True)
eval = data_provider.LanguageDataset("bias_data_reduced/biased.word.test", batch_size=1, remove_stopwords=True)

def count_class_occurances(dataset):
    biased_count = 0
    unbiased_count = 0

    embedder = model.get_embedder()
    dataset = iter(dataset)
    for values, targets in dataset:
        targets = embedder.update_targets(values, targets)[0][1:-1]

        for class_label in targets:
            if class_label == 0:
                unbiased_count += 1
            elif class_label == 1:
                biased_count += 1
            else:
                print("unexpected value in count_class_occurances")

    ratio = unbiased_count / biased_count
    print("Number of biased labels: {}\n Number of unbiased labels: {}\n Ratio: {:.4f}".format(biased_count, unbiased_count, ratio))

"""
Including stopwords
Shows that there are a little under 20 unbiased labels for each biased label
Training dataset:
Number of biased labels: 77534
Number of unbiased labels: 1494766
Ratio: 19.2788

Evaluation dataset:
Number of biased labels: 1395
Number of unbiased labels: 27354
Ratio: 19.6086
"""
""" 
print("Training dataset:")
count_class_occurances(train)
print()
print("Evaluation dataset:")
count_class_occurances(eval)
"""


"""
Removing stopwords
Shows that there are a little under 20 unbiased labels for each biased label
Training dataset:
Number of biased labels: 77534
Number of unbiased labels: 1494766
Ratio: 19.2788

Evaluation dataset:
Number of biased labels: 1395
Number of unbiased labels: 27354
Ratio: 19.6086
"""

print("Training dataset:")
count_class_occurances(train)
print()
print("Evaluation dataset:")
count_class_occurances(eval)