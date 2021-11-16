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
    epochs = 5
    batch_size = 128
    filepath= "Weight-File-{epoch:02d}-{loss:.4f}.hdf5"

    def __init__(self, filepath, checkpoint_folder):
        self.raw_text = self.read_and_return_txt(filepath)[:500]
        self.char_to_num = self.num_char_mapping(reverse=False)
        self.num_to_char = self.num_char_mapping(reverse=True)

        self.n_chars = len(self.raw_text)
        self.vocab_size = len(self.char_to_num)

        self.X, self.y = self.prepare_x_y()
        self.model = None

        self.checkpoint_folder = checkpoint_folder


    def read_and_return_txt(self, path):
        file = open(path, 'r').read()
        text = ' '.join(file.split())
        pattern = re.compile(r"\[(.*?)\]")
        text = re.sub(pattern, '', text)
        return ''.join([i.lower() for i in text if i not in "!?.,:;'"])


    def num_char_mapping(self, reverse):
        characters = sorted(list(set(self.raw_text)))
        mapping = {char : num for num, char in enumerate(characters)}
        if reverse == False:
            return mapping
        return {num : char for char, num in mapping.items()}


    def prepare_x_y(self):
        x_data, y_data = [], []
        chunk_size = 100

        for i in range(0, self.n_chars-chunk_size, 1):
            input_sequence = self.raw_text[i : i + chunk_size]
            output_sequence = self.raw_text[i + chunk_size]
            x_data.append([self.char_to_num[char] for char in input_sequence])
            y_data.append(self.char_to_num[output_sequence])

        x_data  = np.reshape(x_data, (len(x_data), chunk_size, 1))
        x_data = x_data / float(self.vocab_size)
        y_data = np_utils.to_categorical(y_data)

        return x_data, y_data


    def build_model(self):
        model = Sequential()
        model.add(LSTM(256, input_shape=(self.X.shape[1], self.X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(self.y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model

    def train_model(self, keep_best_only):
        filepath = os.path.join(self.checkpoint_folder,
                                'Epoch-{epoch}-{loss:.2f}.hdf5')
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        self.model.fit(self.X,
                       self.y,
                       batch_size=LSTM_Text_Prediction.batch_size,
                       epochs=LSTM_Text_Prediction.epochs,
                       callbacks = [checkpoint],
                       verbose=2)

        if keep_best_only:
            self.delete_old_checkpoints()


    def delete_old_checkpoints(self):
        files = sorted([f for f in os.listdir(self.checkpoint_folder) if f.endswith('.hdf5')])[:-1]
        for f in files:
            os.remove(os.path.join(self.checkpoint_folder, f))
            logging.info('Removed: {}'.format(f))



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    path_to_directory = '/Users/jamesclare/Documents/Python/Machine Learning/TextSequencing'
    path_to_text_file = '/Users/jamesclare/Documents/Python/Machine Learning/Data/Alice.txt'

    LSTM_Model = LSTM_Text_Prediction(path_to_text_file, path_to_directory)
    LSTM_Model.prepare_x_y()
    LSTM_Model.build_model()
    #LSTM_Model.train_model(keep_best_only=True)

    #LSTM_Model.delete_old_checkpoints()
