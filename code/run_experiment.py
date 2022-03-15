import models
import data_provider
import os

# Add the path to the folder containing biased.word.train and biased.word.test
PATH_TO_DATA = ""
# Scales the effect of the classes so that the far more frequent unbiased label is not more significant than the biased label
# Note these weights are normalised, so you just need the desired ratio
CLASS_WEIGHTS = [1,20]
CLASS_WEIGHTS_NO_STOPWORDS = [1,13]
NUM_EPOCHS = 100

training_data = os.path.join(PATH_TO_DATA, "biased.word.train")
evaluation_data = os.path.join(PATH_TO_DATA, "biased.word.test")

# Training and testing dataset, to be applied in each experiment
# Note I believe I have designed these so that they can be reused
train = data_provider.LanguageDataset(training_data, batch_size=32)
eval = data_provider.LanguageDataset(evaluation_data, batch_size=32)
# Variations on the training and testing datasets where the "stopwords" have been removed. 
# These are words which provide little information to the sentence, some examples are as, is, so, to
train_no_stopwords = data_provider.LanguageDataset(training_data, batch_size=32, remove_stopwords=True)
eval_no_stopwords  = data_provider.LanguageDataset(evaluation_data, batch_size=32, remove_stopwords=True)

# Experiment 1: Simple bert model (no lstm or rnn, just a linear layer at the end)
models.BiasDetector(0, models.BertEmbedding, "BERT_only", num_layers=0, class_weights=CLASS_WEIGHTS).run_experiments(train, eval, num_epochs=NUM_EPOCHS)

# Experiment 2: single layer LSTM after the BERT embedding
models.BiasDetector(255, models.BertEmbedding, "single_layer_lstm", num_layers=1, class_weights=CLASS_WEIGHTS).run_experiments(train, eval, num_epochs=NUM_EPOCHS)

# Experiment 3: multi layer LSTM after the BERT embedding
models.BiasDetector(255, models.BertEmbedding, "single_layer_lstm", num_layers=32, class_weights=CLASS_WEIGHTS).run_experiments(train, eval, num_epochs=NUM_EPOCHS)

# Experiments 4, 5 and 6: Rerun 1, 2 and 3 but without stopwords.
models.BiasDetector(0, models.BertEmbedding, "BERT_only", num_layers=0, class_weights=CLASS_WEIGHTS_NO_STOPWORDS).run_experiments(train, eval, num_epochs=NUM_EPOCHS)
models.BiasDetector(255, models.BertEmbedding, "single_layer_lstm", num_layers=1, class_weights=CLASS_WEIGHTS_NO_STOPWORDS).run_experiments(train, eval, num_epochs=NUM_EPOCHS)
models.BiasDetector(255, models.BertEmbedding, "single_layer_lstm", num_layers=32, class_weights=CLASS_WEIGHTS_NO_STOPWORDS).run_experiments(train, eval, num_epochs=NUM_EPOCHS)