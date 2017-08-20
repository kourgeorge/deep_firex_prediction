__author__ = 'gkour'

from ExDataLoader import ExDataLoader
import tensorflow as tf
from Config import Config
from Network import Network

import matplotlib.pyplot as plt


def train():
    config = Config()
    train_data_loader = ExDataLoader(config.train_data_path, config.window_size, config.forecast_horizon,
                                     config.batch_size, config.num_rows)
    val_data_loader = ExDataLoader(config.val_data_path, config.window_size, config.forecast_horizon, config.batch_size, config.num_rows)

    network = Network(config)

    with tf.Session() as sess:
        network.train(sess, train_data_loader, val_data_loader, config.num_epochs, model_path=config.model_path)


def predict():
    config = Config()
    val_data_loader = ExDataLoader(config.train_data_path, config.window_size, config.forecast_horizon, config.batch_size,
                                   3000)

    network = Network(config)
    with tf.Session() as sess:
        val_data_loader.initialize(shuffle=True)
        sequences, labels = val_data_loader.get_all_data()

        last_value_sequence = [sequence[-1] for sequence in sequences]

        predictions = network.predict(sess, sequences, config.model_path)
        # plt.plot(range(1, len(labels)+1), predictions, 'r--', labels, 'b--', last_value_sequence, 'g--')


        for i in range(1, 20):
            before = sequences[i]
            actual = labels[i]
            prediction = predictions[i]
            plt.plot(range(1, config.window_size + 1), before, 'r--')
            plt.hold(True)
            plt.plot(range(config.window_size, config.window_size + config.forecast_horizon), actual, 'b^-')
            plt.plot(range(config.window_size, config.window_size + config.forecast_horizon), prediction, 'g^')

            plt.ylabel('Euro USD')
            plt.show()


#train()
predict()