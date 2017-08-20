__author__ = 'gkour'
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import numpy as np


class Network:
    def __init__(self, config):

        self.sequences = tf.placeholder(tf.float32, shape=[None, config.window_size])
        self.sequences_labels = tf.placeholder(tf.float32, shape=[None, config.forecast_horizon])

        # define the loss metric
        metric = tf.losses.absolute_difference

        inputs = tf.expand_dims(self.sequences, axis=2)

        lstm_cell = LSTMCell(num_units=config.rnn_width, state_is_tuple=True)
        multi_rnn = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_lstm_layers, state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn, inputs=inputs, dtype=tf.float32,
                                           time_major=False)  # sequence_length=self.sequences_length
        #rnn_output = Network._get_last_output(outputs)
        rnn_output = state[0][1]
        layer1 = tf.layers.dense(inputs=rnn_output, units=200, activation=tf.nn.relu, use_bias=True)
        layer2 = tf.layers.dense(inputs=layer1, units=100, activation=tf.nn.relu, use_bias=True)

        # FC instead of LSTM
        layer1 = tf.layers.dense(inputs=self.sequences, units=200, activation=tf.nn.relu, use_bias=True)
        layer2 = tf.layers.dense(inputs=layer1, units=50, activation=tf.nn.relu, use_bias=True)


        self._predictions = tf.layers.dense(inputs=layer2, units=config.forecast_horizon, use_bias=True)

        #self._predictions = tf.Print(self._predictions, [self.sequences_labels], message="This labels are: ", summarize=5)
        #self._predictions = tf.Print(self._predictions, [self._predictions], message="This predictions are: ", summarize=5)

        self._loss = metric(self.sequences_labels, self._predictions, reduction=tf.losses.Reduction.SUM)

        self._optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self._loss)

        batch_size = tf.shape(self.sequences)[0]
        last_value_reshaped = tf.reshape(self.sequences[:, -1], shape=(batch_size, 1))
        last_value_repeated_mat = tf.tile(last_value_reshaped, [1, config.forecast_horizon])

        self._last_value_mse = metric(self.sequences_labels, last_value_repeated_mat, reduction=tf.losses.Reduction.SUM)

        self._saver = tf.train.Saver()

    def train(self, sess, train_data, val_data, num_epochs, model_path=None):

        # Initialize the network variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            cost_epoch = []
            train_data.initialize()
            for batch in range(train_data.get_num_batches()):
                batch_sequences, batch_labels = train_data.next_batch()

                _, cost_batch = sess.run([self._optimizer, self._loss],
                                         feed_dict={self.sequences: batch_sequences,
                                                    self.sequences_labels: batch_labels})

                cost_epoch.append(cost_batch)

            # Run test on the validation set
            metric = self.test(sess, train_data)
            print("Epoch: " + str(epoch))
            print("\tTrain cost: " + str(np.sum(cost_epoch)))
            print("\tTrain mean squared error: " + str(metric))

            # Implementing Early stopping, if the model is better than the naive last value model, stop.
            #if metric[0] <= 0.0006:
            #    break

            #metric, l_v_mse = self.test(sess, test_data=val_data)
            #print("\tValidation mean squared error: " + str(metric))
            #print("\tLast Value mean squared error: " + str(l_v_mse))

        if model_path is not None:
            self._saver.save(sess, model_path)

        return np.mean(cost_epoch)

    def test(self, sess, test_data):
        test_data.initialize()

        batch_sequences, batch_labels = test_data.get_all_data()
        mean_squared_error, last_value_mse = sess.run([self._loss, self._last_value_mse],
                                               feed_dict={self.sequences: batch_sequences,
                                                          self.sequences_labels: batch_labels})

        return mean_squared_error, last_value_mse

    def predict(self, sess, sequence, model_path=None):
        if model_path is not None:
            self._saver.restore(sess, model_path)

        return sess.run(self._predictions, feed_dict={self.sequences: sequence})

    @staticmethod
    def _get_last_output(outputs):
        last_index = tf.shape(outputs)[1] - 1
        outputs_rs = tf.transpose(outputs, [1, 0, 2])
        return tf.nn.embedding_lookup(outputs_rs, last_index)
