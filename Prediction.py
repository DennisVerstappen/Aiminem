
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sys
import pickle

#import keras 
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop
import h5py


# In[ ]:


# load the model
folder = "files/"
model = load_model(folder + 'Aiminem_model.h5')  # creates a HDF5 file 'my_model.h5'

# load chars
infile = open(folder + 'chars.pkl','rb')
chars = pickle.load(infile)
infile.close()

# load char_indices
infile = open(folder + 'char_indices.pkl','rb')
char_indices = pickle.load(infile)
infile.close()

# load indices_char
infile = open(folder + 'indices_char.pkl','rb')
indices_char = pickle.load(infile)
infile.close()

# set lyric lenght
LYRIC_LENGTH = 80


# In[15]:


sentence = "Yo, his palms are sweaty "
sequence_length = 20
sentence = sentence[-19:].lower()
generated = sentence
temperature = 0.1

for i in range(LYRIC_LENGTH):
    x = np.zeros((1, sequence_length, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
    
    # do prediction
    predictions = model.predict(x, verbose=0)[0]
    
    # get char with highest probability 
    preds = np.asarray(predictions).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    next_index = np.argmax(probas)
    
    next_char = indices_char[next_index]
    generated += next_char
    sentence = sentence[1:] + next_char


print(generated)
print(len(generated))
