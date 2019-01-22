
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

#import keras 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adam
import h5py

# check if tensorflow is using gpu
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

unwanted_chars = ['!', '"', '%', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[', ']', '{', '}']

# create function to remove unwanted characters in text. Makes training easier
def remove_chars(text, chars):
    for char in chars:
        if text:
            text_cleaned = text.replace(char, "")
            text = text_cleaned
    return text


# load all songs
songs = pd.read_csv('files/eminem_lyrics.csv', sep=';')

text =""

# merge all the lyrics together into one huge string
for index, row in songs['lyrics'].iteritems():
    stripped_text = remove_chars(str(row).lower(), unwanted_chars)
    text = text + stripped_text


# find all the unique chracters
chars = sorted(list(set(text)))
print('total chars:', len(chars))

# create a dictionary mapping chracter-to-index
char_indices = dict((c, i) for i, c in enumerate(chars))

# create a dictionary mapping index-to-chracter
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text into sequences
maxlen = 20
step = 1 # step size at every iteration
sentences = [] # list of sequences
next_chars = [] # list of next chracters that our model should predict

# iterate over text and save sequences into lists
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
    
# create empty matrices for input and output sets 
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# iterate over the matrices and convert all characters to numbers
# basically Label Encoding process and One Hot vectorization
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# create sequential network, because we are passing activations
# down the network
model = Sequential()

# add LSTM layer
model.add(LSTM(256, input_shape=(maxlen, len(chars)) ))
model.add(Dropout(0.2))

# add Softmax layer to output one character
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# compile the model and pick the loss and optimizer
model.compile(loss='categorical_crossentropy', optimizer='Adam')

# train the model
history = model.fit(x, y,  validation_split=0.1, batch_size=128, epochs=75, verbose=1)


# save the model
model.save('files/Aiminem_model.h5')  # creates a HDF5 file 'my_model.h5'

objects_vars = {"chars": chars, "char_indices": char_indices, "indices_char": indices_char, "history": history}

for name, object_var in objects_vars.items():
    # save
    name_file = "files/" + name + ".pkl"
    outfile = open(name_file,'wb')
    pickle.dump(object_var, outfile)
    outfile.close()