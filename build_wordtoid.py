from collections import Counter
from chunker_io import read_data_from_file

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
    training_data = read_data_from_file('trainfilename')
    word_counts = get_word_counts(training_data)
    write_word_ix(word_counts)
    
if __name__ == '__main__':
    main()
            