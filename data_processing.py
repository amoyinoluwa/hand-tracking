import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        self.all_data = []
        self.all_labels = []

    def preprocess_data(self, file_path):
        scaler = MinMaxScaler()
        data_frame = pd.read_csv(file_path).iloc[:, :-1]
        scaled_data = scaler.fit_transform(data_frame)
        return scaled_data

    def read_path(self, path):
        data = []
        for file in os.listdir(path):
            curr_array = self.preprocess_data(os.path.join(path, file))
            if curr_array.shape[0] >= 30:
                data.append(np.array(curr_array[:30]))
        return data

    def process_data(self):
        for i in range(1, 4):
            curr_data = self.read_path(f'../data/Gesture-{i}')
            self.all_data.extend(curr_data)
            self.all_labels.extend([f'gesture{i}' for _ in range(len(curr_data))])
        features = np.array(self.all_data)
        labels = np.array(self.all_labels)
        features = features.reshape((*features.shape, 1))
        return features, labels