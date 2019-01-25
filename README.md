# Concept Chunker

### Overall Project Goal
Find out the relationships between concepts (ie. Addition, Arithmetic, Calculus) in MOOC courses.

### Role of Chunker
To find relationships between concepts in a course, we must first figure out which words should be grouped together to form concepts. We train the model on textbook data where we have gold standard concepts from the index and test on lecture data where a CS graduate student has manually annotated the data.

### Model Design
Model uses IOB tagging to determine where multiword concept phrases start and end
1 LSTM Hidden Layer (PyTorch)

### Results
TBD

### Installation
1. Install torch
2. Install sklearn
3. Install spacy
4. python3 -m spacy download en_core_web_sm

### How to Run
1. python3 split_train_data.py (to split textbook data into train and development)
2. python3 build_wordtoid.py (To build mapping from words to indices in embedding layer)
3. python3 train.py to train model
4. Test on lecture data

### Example:
1. python3 split_train_data.py -i zhai_tb_iob_tags.txt -t zhai_tb_train.txt -d zhai_tb_dev.txt
2. python3 build_wordtoid.py -i zhai_tb_train.txt -o zhai_tb_wordids.txt
3. python3 train.py -m zhai_tb_model.pt -w zhai_tb_wordids.txt -t zhai_tb_train.txt -d zhai_tb_dev.txt
4. python3 eval.py -t zhai_tb_dev.txt -m zhai_tb_model.pt -w zhai_tb_wordids.txt

### TODO:
- Test bidirectional model
- Test out different embedding and hidden dimensions
- Test out batch training
- Use unk tags for least frequent words?