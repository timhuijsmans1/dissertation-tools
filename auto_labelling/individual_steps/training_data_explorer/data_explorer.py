import os
import json
import numpy as np

class dataExplorer:
    def data_reader(self, training_data_path):
        neg_count = 0
        pos_count = 0
        count = 0
        with open(training_data_path, 'r') as f:
            for line in enumerate(f):
                count += 1
        print(count)
        return

if __name__ == "__main__":
    TRAINING_DATA_PATH = '../data/preprocessed_data/preprocessed.txt'
    explorer = dataExplorer()
    explorer.data_reader(TRAINING_DATA_PATH)