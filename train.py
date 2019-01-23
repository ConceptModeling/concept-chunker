#Based on :
#Author: Robert Guthrie
#http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chunker_io import read_data_from_file
from lstm_tagger import LSTMTagger
from build_wordtoid import read_word_ix

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import argparse

torch.manual_seed(37)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, 0) for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def print_metrics(model, testing_data, word_to_ix, tag_to_ix):
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
    #test_to_file(testing_data, test_pred, ix_to_tag)

def main():
    parser = argparse.ArgumentParser(description=
        '''Trains the model''')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--print_metrics', type=bool, default=True)

    parser.add_argument('--model_filename', '-m')

    parser.add_argument('--word_ix_filename', '-w')
    parser.add_argument('--train_filename', '-t')
    parser.add_argument('--dev_filename', '-d')
    args = parser.parse_args()

    tag_to_ix = {'B': 0, 'I': 1, 'O': 2}
    ix_to_tag = {0: 'B', 1: 'I', 2: 'O'}
    word_to_ix = read_word_ix(args.word_ix_filename)

    training_data = read_data_from_file(args.train_filename)
    testing_data = read_data_from_file(args.dev_filename)

    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix) + 1, len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if args.train:
        for epoch in range(1, 10): #9 Epochs
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
            if args.print_metrics:
                print("Epoch:", epoch)
                print_metrics(model, testing_data, word_to_ix, tag_to_ix)
        torch.save(model.state_dict(), args.model_filename)
    else:
        model.load_state_dict(torch.load(args.model_filename))
        print_metrics(model, testing_data, word_to_ix, tag_to_ix)

if __name__ == '__main__':
    main()