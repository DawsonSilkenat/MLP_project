from sys import stdout
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, Trainer, TrainingArguments
from transformers import BertTokenizer, BertTokenizerFast
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available


# define test data example
test_seq = ['this is a test example.', 'this will be replaced by real data point later.']

# some helper functions

def save_obj(obj, name ):
    with open('../obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)



set_seed(1)

'''
load BERT tokenizer
NB: 
    the BERT tokenizer contains the preprocessing part
    num_label can be adjusted later
    here we use BertForSequenceClassification, can also try AutoModel
'''

# config = BertConfig.from_pretrained( 'bert-base-uncased', output_hidden_states=True)    
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# set parameters
max_seq_len = 255

# tokenize and encode sequences in the training set
train_encodings = tokenizer(
    test_seq,
    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
    max_length = max_seq_len,
    pad_to_max_length=True,
    return_attention_mask = True,
    return_tensors = 'pt',
    truncation=True,
    return_token_type_ids=True
)

stdout.write(str(train_encodings))
