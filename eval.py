import torch

from chunker_io import read_data_from_file
from lstm_tagger import LSTMTagger
from build_wordtoid import read_word_ix
from train import prepare_sequence, print_metrics

import argparse

torch.manual_seed(37)

def main():
    parser = argparse.ArgumentParser(description='Evaluates the Model')
    parser.add_argument('--test_filename', '-t')
    parser.add_argument('--model_filename', '-m')
    parser.add_argument('--word_ix_filename', '-w')
    args = parser.parse_args()

    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32

    tag_to_ix = {'B': 0, 'I': 1, 'O': 2}
    ix_to_tag = {0: 'B', 1: 'I', 2: 'O'}
    word_to_ix = read_word_ix(args.word_ix_filename)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix) + 1, len(tag_to_ix), bidirectional=True)
    model.load_state_dict(torch.load(args.model_filename))
    model.eval()

    testing_data = read_data_from_file(args.test_filename)
    print_metrics(model, testing_data, word_to_ix, tag_to_ix)

if __name__ == '__main__':
    main()
