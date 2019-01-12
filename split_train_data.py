from chunker_io import read_data_from_file

def train_to_file(training_data, filename):
    with open(filename, "w") as outfile:
        for sentence, tags in training_data:
            for word, tag in zip(sentence, tags):
                outfile.write(word + " " + tag + "\n")
            outfile.write(". O\n")


def main():
    training_data, testing_data = train_test_split(
        read_data_from_file(TAGGED_TEXTBOOK), random_state=37
    )

if __name__ == '__main__':
    main()