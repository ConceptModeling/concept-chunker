from chunker_io import read_data_from_file
import argparse
from sklearn.model_selection import train_test_split

def train_to_file(training_data, filename):
    #print(training_data)
    with open(filename, "w") as outfile:
        for sentence, tags in training_data:
            for word, tag in zip(sentence, tags):
                outfile.write(word + "\t" + tag + "\n")
            outfile.write(".\tO\n")


def main():
    parser = argparse.ArgumentParser(description=
        '''Splits the tagged input file into train and development datasets.''')
    parser.add_argument('--input_filename', '-i')
    parser.add_argument('--train_filename', '-t')
    parser.add_argument('--dev_filename', '-d')
    args = parser.parse_args()

    training_data, dev_data = train_test_split(
        read_data_from_file(args.input_filename), random_state=37
    )
    train_to_file(training_data, args.train_filename)
    train_to_file(dev_data, args.dev_filename)
if __name__ == '__main__':
    main()