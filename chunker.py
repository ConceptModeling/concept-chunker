#Based on :
#Author: Robert Guthrie
#http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chunker_io import read_data_from_file, read_data_from_dir
from lstm_tagger import LSTMTagger

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, 0) for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def has_concept(data):
    new_data = []
    for sent, tags in data:
        if 'B' in tags:
            new_data.append((sent, tags))
    return new_data

training_data, testing_data = train_test_split(has_concept(read_data_from_file('ml-tagged-files/Bishop - Pattern Recognition And Machine Learning - Springer  2006.txt')), random_state=42)

print('Train Size:', len(training_data))
print('Test Size:', len(testing_data))

word_to_ix = {}
tag_to_ix = {'B': 0, 'I': 1, 'O': 2}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) + 1

EMBEDDING_DIM = 32
HIDDEN_DIM = 32

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix) + 1, len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def print_metrics():
    #http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    test_pred = []
    test_true = []
    for sentence, tags in testing_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        tag_scores = model(sentence_in)
        targets = prepare_sequence(tags, tag_to_ix)
        _, predicted_tags = torch.max(tag_scores, 1, keepdim=True)
        test_pred += predicted_tags.data.tolist()
        test_true += targets.data.tolist()

    print(metrics.confusion_matrix(test_true, test_pred))
    print('Acc:', metrics.accuracy_score(test_true, test_pred))
    print('F1:', metrics.f1_score(test_true, test_pred, average='macro'))
    print('Precision:', metrics.precision_score(test_true, test_pred, average='macro'))
    print('Recall:', metrics.recall_score(test_true, test_pred, average='macro'))

for epoch in range(8):
    i = 0
    for sentence, tags in training_data:
        if i % 100 == 0:
            print(i)
        i+= 1
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
    print("Epoch:", epoch)
    print_metrics()
