import torch

from chunker_io import read_data_from_file_no_tags, write_tagged_data_to_file
from lstm_tagger import LSTMTagger
from build_wordtoid import read_word_ix
from train import prepare_sequence, print_metrics

import argparse

torch.manual_seed(37)

def tagged_data(model, testing_data, word_to_ix, ix_to_tag):
    model.eval()
    #http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    tagged_sentences = []
    for sentence in testing_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        tag_scores = model(sentence_in)
        _, predicted_tags = torch.max(tag_scores, 1, keepdim=True)
        tagged_sentences.append(list(zip(
            sentence,
            [ix_to_tag[ixl[0]] for ixl in predicted_tags.data.tolist()])) + [('.', 'O')])
    return tagged_sentences


def main():
    parser = argparse.ArgumentParser(description='Evaluates the Model')
    parser.add_argument('--test_filename', '-t')
    parser.add_argument('--model_filename', '-m')
    parser.add_argument('--word_ix_filename', '-w')
    parser.add_argument('--output_filename', '-o')

    args = parser.parse_args()

    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32

    tag_to_ix = {'B': 0, 'I': 1, 'O': 2}
    ix_to_tag = {0: 'B', 1: 'I', 2: 'O'}
    word_to_ix = read_word_ix(args.word_ix_filename)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix) + 1, len(tag_to_ix), bidirectional=True)
    model.load_state_dict(torch.load(args.model_filename))

    testing_data = read_data_from_file_no_tags(args.test_filename)
    tag_data = tagged_data(model, testing_data, word_to_ix, ix_to_tag)
    write_tagged_data_to_file(tag_data, args.output_filename)
    
if __name__ == '__main__':
    main()
