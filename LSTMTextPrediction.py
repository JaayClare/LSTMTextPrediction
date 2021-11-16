from string import punctuation
from keras.utils import np_utils
import numpy as np
import os
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import logging


class LSTM_Text_Prediction:
    def __init__(self, filepath, path_to_model):
        self.raw_text = self.read_and_return_txt(filepath)
        self.character_to_num = self.character_to_integer()
        self.num_to_character = self.integer_to_character()

        self.n_chars = len(self.raw_text)
        self.vocab_size = len(self.character_to_num)

        self.model = None
        self.path_to_model = path_to_model


    def read_and_return_txt(self, path):
      file = open(path, 'r').read()
      text = ' '.join(file.split())
      pattern = re.compile(r"\[(.*?)\]")
      text = re.sub(pattern, '', text)
      text = ''.join([i.lower() for i in text if i not in "!?.,:;'"])
      return text


    def integer_to_character(self):
        characters = sorted(list(set(self.raw_text)))
        return {char : num for char, num in enumerate(characters)}


    def character_to_integer(self):
        characters = sorted(list(set(self.raw_text)))
        return {num: char for char, num in enumerate(characters)}


    def build_model_for_prediction(self):
        model = Sequential()
        model.add(LSTM(256, input_shape=(100, 1)))
        model.add(Dropout(0.2))
        model.add(Dense(23, activation='softmax'))
        model.load_weights(self.path_to_model)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model


    def generate_chars(self):
        X,y = [], []

        for i in range(0, self.n_chars - 100):
            seq_in = self.raw_text[i:i + 100]
            seq_out = self.raw_text[i + 100]

            X.append([self.character_to_num[char] for char in seq_in])

        start = np.random.randint(0, len(X)-1)
        txt_pattern = X[start]

        for i in range(100):
            x = np.reshape(txt_pattern, (1, len(txt_pattern), 1))
            x = x / float(self.vocab_size)

            prediction = self.model.predict(x, verbose=0)
            prediction_idx = np.argmax(prediction)

            result = self.num_to_character[prediction_idx]
            txt_pattern.append(prediction_idx)
            txt_pattern = txt_pattern[1:len(txt_pattern)]


        print(''.join([self.num_to_character[i] for i in txt_pattern]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    path_to_model = '/Users/jamesclare/Documents/Python/Machine Learning/TextSequencing/Epoch-7-2.81.hdf5'
    path_to_text_file = '/Users/jamesclare/Documents/Python/Machine Learning/Data/Alice.txt'

    LSTM_Model = LSTM_Text_Prediction(path_to_text_file, path_to_model)
    LSTM_Model.build_model_for_prediction()
    LSTM_Model.generate_chars()
