import torch

from chunker_io import read_data_from_file
from lstm_tagger import LSTMTagger
from build_wordtoid import read_word_ix
from train import prepare_sequence, print_metrics
import re
import argparse
import spacy

torch.manual_seed(37)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

#from iob_tagger_textbook.py
def preprocess(words):
    #Remove Line breaks
    words = re.sub(r'''-\n''',r"",words)
    #Remove weird fi character
    words = re.sub(r'ﬁ',r'fi', words)
    #Normalize aprostrophes
    words = re.sub(r'''’''',r"'",words)
    #Remove non alphabetic characters
    words = re.sub(r'''[^a-zA-Z\u0391-\u03A9\u03B1-\u03C9\ \-\.]''',r''' ''',words)
    #Collapse sequences of whitespace
    words = re.sub(r'''\s+''',r''' ''',words)
    #Normalize to lowercase
    return [token.lemma_ for token in nlp(words.strip())]

def main():
    tag_to_ix = {'B': 0, 'I': 1, 'O': 2}
    ix_to_tag = {0: 'B', 1: 'I', 2: 'O'}

    parser = argparse.ArgumentParser(description='Get Concept Embeddings')
    parser.add_argument('--model_filename', '-m')
    parser.add_argument('--word_ix_filename', '-w')
    parser.add_argument('--concept_filename', '-c')

    args = parser.parse_args()

    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32

    word_to_ix = read_word_ix(args.word_ix_filename)
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix) + 1, len(tag_to_ix), bidirectional=True, log=True)
    model.load_state_dict(torch.load(args.model_filename))
    
    with open(args.concept_filename) as infile:
        for line in infile:
            model.hidden = model.init_hidden()
            print(line.strip())
            model(prepare_sequence(preprocess(line), word_to_ix))
            print()
if __name__ == '__main__':
    main()
