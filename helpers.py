import time
import math
import string
import random
import os
import torch
import glob
import argparse
import pickle
import unicodedata
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)

# don't change
parser.add_argument('--seq_length', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--rnn_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)

# changing these seems fine
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=2e-3)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--output', '-o', type=str, default='checkpoints')
parser.add_argument('--seed', type=str, default=random.choice(string.printable))
# ^ if u don't specify a seed it will just pick a random character

# for generating
# these are in the same file bc if u split them up it doesn't like it
parser.add_argument('--temperature', type=float, default=0.8, help='for generate')
parser.add_argument('--sample_len', type=int, default=500, help='for generate')
parser.add_argument('--checkpoint', '-c', type=str, help='for generate')
parser.add_argument('--charfile', '-f', type=str, help='for generate')
parser.add_argument('--concatenate', type=int, default=0, help='for generate')
args = parser.parse_args()

def findFiles(path):
    return glob.glob(path)

# randomise runs
torch.manual_seed(np.random.randint(1, 9999))
random_state = np.random.RandomState(np.random.randint(1, 9999))

seq_length = args.seq_length
batch_size = args.batch_size
hidden_size = args.rnn_size
epoch_count = args.max_epochs
n_layers = args.num_layers
lr = args.learning_rate
dropout = args.dropout
filename = args.filename
checkpoint_prepend = os.path.join(args.output, 'checkpoint_')
final_checkpoint_prepend = os.path.join(args.output, 'final_checkpoint_')

# find files in a directory

folder = findFiles(filename + '/*.txt')
sum = 0  # for calculating total length

all_texts = ""

# i don't really understand batching and i don't want to learn so this goes
# through a specified folder and pulls texts from all the .txt files in it
for filename in folder:
    f = open(filename, 'r')  # open file
    content = f.read()  # read file
    length = len(content)  # get length of file
    sum = sum + length  # add up file lengths
    all_texts = all_texts + content  # combine all text files into one text

# this was originally only pulling characters from the text but turns out people
# don't use a lot of characters in writing so now its just pulling all chars
chars = string.printable # get all characters
charfile = filename[0] + '_chars.pkl'
with open(charfile, 'wb') as f:
    pickle.dump(chars, f)

# unicode to ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in chars
    )

all_texts = unicodeToAscii(all_texts)

chars_len = len(chars) # amount of possible characters
char_to_index = {}
index_to_char = {}

for i, c in enumerate(chars):
    char_to_index[c] = i
    index_to_char[i] = c

def index_to_tensor(index):
    tensor = torch.zeros(1, 1).long()
    tensor[0,0] = index
    return tensor

# batching

def chunks(l, n):
    for i in range(0, len(l) - n, n):
        yield l[i:i + n]

if len(content) == 0:
    raise RuntimeError('data not found')

# get random part of text and get a random chunk
