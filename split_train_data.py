from chunker_io import read_data_from_file
import argparse

def train_to_file(training_data, filename):
    with open(filename, "w") as outfile:
        for sentence, tags in training_data:
            for word, tag in zip(sentence, tags):
                outfile.write(word + " " + tag + "\n")
            outfile.write(". O\n")


def main():
    parser = argparse.ArgumentParser(description=
        '''Splits the tagged input file into train and development datasets.''')
    parser.add_argument('--input_filename', '-i')
    parser.add_argument('--train_filename', '-t')
    parser.add_argument('--dev_filename', '-d')
    args = parser.parse_args()

    training_data, testing_data = train_test_split(
        read_data_from_file(args.input_file), random_state=37
    )
    train_to_file(training_data, args.train_filename)
    train_to_file(args.dev_filename, args.train_filename)
if __name__ == '__main__':
    main()