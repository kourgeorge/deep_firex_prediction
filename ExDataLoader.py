__author__ = 'gkour'

import numpy as np
import pandas as pd
import math

class ExDataLoader(object):
    def __init__(self, data_file_path, window_size, prediction_forecast, batch_size=None, num_rows=None):

        self.file_path = data_file_path
        self._window_size = window_size
        self._prediction_forecast = prediction_forecast
        self._batch_size = batch_size
        self._sequences = None
        self._labels = None
        self._batch_num = None
        self.num_rows = num_rows

        self._orig_sequence = ExDataLoader._load_data(self.file_path, self.num_rows)

    def initialize(self, shuffle=True):
        """ Load and shuffle the data set and initialize the batches.
        """
        #This delay s to start the labels after a delay
        delay = 0

        sequence_start_max_index = len(self._orig_sequence) - self._window_size - self._prediction_forecast- delay
        # Shuffle rows
        if shuffle:
            start_indices_perm = np.random.permutation(sequence_start_max_index)
        else:
            start_indices_perm = range(1, sequence_start_max_index)

        sequences = []
        labels = []
        for i in start_indices_perm:
            sequences.append(self._orig_sequence[i:i + self._window_size, 0])
            labels.append(self._orig_sequence[i + self._window_size + delay: i + self._window_size + delay + self._prediction_forecast, 0])

        self._sequences = sequences
        #self._labels = np.reshape(labels, newshape=(-1, self._prediction_delay))
        self._labels = labels

        self._batch_num = 0

    def next_batch(self):
        """ Returns the next batch of samples.
        :return: a three place tuple containing thesis sequence, fact sequence and label
        """
        if self._batch_size is None:
            return self._sequences, self._labels

        start_ind = self._batch_size * self._batch_num
        end_ind = min(start_ind + self._batch_size, self.num_samples())

        if start_ind >= self.num_samples():
            print("You have reached the end of data, use initialize to start iterating over.")
            return None

        batch_sequences = self._sequences[start_ind:end_ind]
        batch_labels = self._labels[start_ind:end_ind]

        self._batch_num += 1
        return batch_sequences, batch_labels

    def get_all_data(self):
        """ get all the sample set in a single batch.
        """
        return self._sequences, self._labels

    def get_num_batches(self):
        """ The total number of batches depending on the dataset size and the batch size.
        :return: the total number of batches
        """
        if self._batch_size is None:
            return 1

        num_batches = math.ceil(self.num_samples() / self._batch_size)
        return num_batches

    def num_samples(self):
        """ Returns the total number of samples.
        """
        return len(self._orig_sequence) - self._window_size

    def batch_size(self):
        """ Returns the batch size
        """
        return self._batch_size

    @staticmethod
    def _load_data(path, num_rows):
        df = pd.read_csv(path, sep=',', header=None, index_col=False, skiprows = 1000,
                         names=['date', 'time', 'high', 'low', 'open', 'close', 'unk'], nrows=num_rows)
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #return scaler.fit_transform(df.as_matrix(['open']))

        return df.as_matrix(['open'])