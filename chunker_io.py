#ignores last sentence if no period
def read_data_from_file(filename, sep='\t'):
    data = []
    sent_counter = 0
    with open(filename) as infile:
        lines = infile.readlines()
        sent_t = ([], [])
        for line in lines:
            lsplit = line.strip().split(sep)
            if len(lsplit) == 2:
                word, chunk_tag = lsplit
                if word == '.':
                    if sent_t[0]:
                        data.append(sent_t)
                        sent_t = ([], [])
                        sent_counter += 1
                else:
                    sent_t[0].append(word)
                    sent_t[1].append(chunk_tag)
    return data

def read_data_from_file_no_tags(filename):
    data = []
    sent_counter = 0
    with open(filename) as infile:
        lines = infile.readlines()
        sents = []
        for line in lines:
            word = line.strip()
            if word == '.':
                if sents:
                    data.append(sents)
                    sents = []
                    sent_counter += 1
            else:
                sents.append(word)
    return data

def write_tagged_data_to_file(data, filename, sep='\t'):
    sent_counter = 0
    with open(filename, 'w') as outfile:
        for sent in data:
            for word, tag in sent:
                outfile.write(word + sep + tag + '\n')