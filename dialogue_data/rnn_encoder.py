from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
learning_rate = 0.1
hidden_size = 256


class Lang:
    def __init__(self, name="dict"):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {EOS_token:"EOS",SOS_token:"SOS"}
        self.n_words = 2  

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)  
        self.gru = nn.GRU(hidden_size, hidden_size) 

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1) 
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def prepareData():
	lang_dict = Lang()
	sentence = []
	reply = []
	with open("extracted_dialogue.csv","r") as f:
		lines = f.read().strip().split("\n")
		for line in lines:
			sentence.append(line.split(",")[0])
			lang_dict.addSentence(line.split(",")[0])
	return lang_dict, sentence, reply

def getSentenceTensor(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


lang_dict, sentence, reply = prepareData()

encoder = EncoderRNN(lang_dict.n_words, hidden_size).to(device)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
criterion = None

for i in range(10):
	input_tensor = getSentenceTensor(sentence[i])
	target_tensor = getSentenceTensor(reply[i])
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	for i in range(input_length):
		encode_code, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
	print(encode_code, encoder_hidden)
