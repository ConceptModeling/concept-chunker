import os
def read_files(directory):
    sentences = []
    cur_sentence = []
    for filename in os.listdir(directory):
        if os.path.isfile(directory + '/' + filename):
            with open(directory + '/' +  filename) as infile:
                for line in infile:
                    split_line = line.strip().split('\t')
                    word, tag = split_line
                    #print(word, tag)
                    if word == '.':
                        if cur_sentence:
                            sentences.append(cur_sentence)
                            cur_sentence = []
                    else:
                        cur_sentence.append((word, tag))
    print(sentences[:100])
    print(len(sentences))
