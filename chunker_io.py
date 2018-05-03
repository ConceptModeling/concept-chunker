def read_data_from_file(filename, sep=' '):
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

def read_data_from_dir(dir):
    data = []
    sent_counter = 0
    for filename in os.listdir(directory):
        if os.path.isfile(directory + '/' + filename):
            with open(directory + '/' +  filename) as infile:
                lines = infile.readlines()
                sent_t = ([], [])
                for line in lines:
                    word, chunk_tag = line.strip().split()
                    if word == '.':
                        if sent_t[0]:
                            data.append(sent_t)
                            sent_t = ([], [])
                            sent_counter += 1
                    else:
                        sent_t[0].append(word)
                        sent_t[1].append(chunk_tag)
    return data
