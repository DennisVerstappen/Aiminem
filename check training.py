import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

#import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import RMSprop
import h5py

folder = "files/"

# load history
infile = open("files/history.pkl",'rb')
history = pickle.load(infile)
infile.close()

print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
