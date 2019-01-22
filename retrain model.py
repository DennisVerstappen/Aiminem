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
model = load_model(folder + 'Aiminem_model_retrained.h5')  # creates a HDF5 file 'my_model.h5'

# check if tensorflow is using gpu
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# load all songs
songs = pd.read_csv('files/eminem_lyrics.csv', sep=';')

text = ""

songs_df = songs['lyrics'].sample(frac=1).reset_index(drop=True)
print(songs_df)

# merge all the lyrics together into one huge string
for index, row in songs_df.iteritems():
    text = text + str(row).lower()

# find all the unique chracters
chars = sorted(list(set(text)))
print('total chars:', len(chars))

# create a dictionary mapping chracter-to-index
char_indices = dict((c, i) for i, c in enumerate(chars))

# create a dictionary mapping index-to-chracter
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text into sequences
maxlen = 20
step = 1  # step size at every iteration
sentences = []  # list of sequences
next_chars = []  # list of next chracters that our model should predict

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


# train the model
history = model.fit(x, y, batch_size=128, validation_split=0.2, epochs=50, verbose=1, initial_epoch=75)


# save the model
model.save('files/Aiminem_model_retrained.h5')  # creates a HDF5 file 'my_model.h5'


# save history
name_file = "files/" + "history_retrain" + ".pkl"
outfile = open(name_file, 'wb')
pickle.dump(history, outfile)
outfile.close()