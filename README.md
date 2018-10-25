# Concept Chunker
### Overall Project Goal
Find out the relationships between concepts (ie. Addition, Arithmetic, Calculus) in MOOC courses.
### Role of Chunker
To find relationships between concepts in a course, we must first figure out which words should be grouped together to form concepts. 
### Model Design
Model uses IOB tagging to determine where multiword concept phrases start and end
1 LSTM Hidden Layer (PyTorch)
### Results
MOOC Lecture Dataset (Test): Accuracy: .961, Macro F1: .883
