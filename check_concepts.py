# Not currently used, need to modify for new format

def get_concepts_train(training_filename):
    concept_set = set()

    with open(training_filename) as infile:
        cur_concept = []
        for line in infile:
            word, tag = line.strip().split()
            if cur_concept and tag != 'I':
                concept_set.add(" ".join(cur_concept))
                cur_concept = []
            if tag == 'B' or tag == 'I':
                cur_concept.append(word)
        if cur_concept:
            concept_set.add(" ".join(cur_concept))
            if correct_tag_seq(pred_tags):
                found_concept_set.add(" ".join(cur_concept))
    return concept_set

def get_concepts_test(training_filename):
    concept_set = set()
    found_concept_set = set()

    def correct_tag_seq(tag_seq):
        if tag_seq[0] != 'B':
            return False
        for tag in tag_seq[1:]:
            if tag != 'I':
                return False
        return True

    with open(training_filename) as infile:
        cur_concept = []
        pred_tags = []
        for line in infile:
            word, true_tag, pred_tag = line.strip().split()
            if cur_concept and true_tag != 'I':
                concept_set.add(" ".join(cur_concept))
                if correct_tag_seq(pred_tags) and pred_tag != 'I':
                    found_concept_set.add(" ".join(cur_concept))
                cur_concept = []
                pred_tags = []
            if true_tag == 'B' or true_tag == 'I':
                cur_concept.append(word)
                pred_tags.append(pred_tag)
        if cur_concept:
            concept_set.add(" ".join(cur_concept))
            if correct_tag_seq(pred_tags):
                found_concept_set.add(" ".join(cur_concept))
    return concept_set, found_concept_set

train_concept_set = get_concepts_train('concept_train_data.txt')
_, found_test_concept_set = get_concepts_test('concept_test_data.txt')

print(train_concept_set)
print()
print(found_test_concept_set)
print()
print(found_test_concept_set-train_concept_set)
