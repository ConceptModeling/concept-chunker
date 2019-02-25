import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(37)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=False, log=False):
        super(LSTMTagger, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        if self.bidirectional:
            n = 2
        else:
            n = 1 
        self.hidden2tag = nn.Linear(n*hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        self.should_log=log

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.bidirectional:
            n = 2
        else:
            n = 1
        return (autograd.Variable(torch.zeros(n, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(n, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        if self.should_log:
            # print(self.hidden)
            # print(self.hidden[1].data.numpy())
            print(self.hidden[1].data.numpy()[0][0].tolist(), self.hidden[1].data.numpy()[1][0].tolist())
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
