from collections import Counter
from chunker_io import read_data_from_file
import argparse

def get_word_counts(training_data):
    counter = Counter()
    for sent, tags in training_data:
        for word in sent:
            counter[word] += 1
    return counter

def write_word_ix(word_counts, output_file, vocab_size=500000):
    word_tuples = word_count.most_common(vocab_size) # Does this work if vocab_size > len(word_count)
    with open(output_file, 'w') as outfile:
        for tup in word_tuples:
            outfile.write(tup[0] + '\n')

def read_word_ix(input_file):
    word_to_ix = {}
    index = 0
    with open(input_file) as infile:
        for word in file:
            word_to_ix[word] = index
            index += 1
    return word_to_ix

def main():
    parser = argparse.ArgumentParser(description=
        '''Makes a file with each stemmed word from the Vocabulary.
        Line number - 1 corresponds to the index in the embedding table.''')
    parser.add_argument('--input_filename', '-i')
                        #dest='accumulate',
                        #default=max,
                        #help='sum the integers (default: find the max)')
    parser.add_argument('--output_filename', '-o')
    args = parser.parse_args()

    training_data = read_data_from_file(args.input_file)
    word_counts = get_word_counts(training_data)
    write_word_ix(word_counts, args.output_file)
    
if __name__ == '__main__':
    main()
            