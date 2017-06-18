__author__ = 'gkour'


class Config:
    def __init__(self):
        self.learning_rate = 0.001
        self.rnn_width = 300
        self.num_epochs = 600
        self.batch_size = 3600
        self.window_size = 8
        self.prediction_delay = 1
        self.num_lstm_layers = 1
        #self.train_data_path = "./data/eurusdpre2015.csv"
        #self.val_data_path = "./data/eurusd2015.csv"

        self.train_data_path = "./data/dev.csv"
        self.val_data_path = "./data/dev.csv"


        self.model_path = "./model/model.ckpt"
        self._results_file_path = "./data/classification.txt"


