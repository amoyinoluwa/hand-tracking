import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import asyncio
import websockets
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            curr_data = self.read_path(f'./data/Gesture-{i}')
            self.all_data.extend(curr_data)
            self.all_labels.extend([f'gesture{i}' for _ in range(len(curr_data))])
        features = np.array(self.all_data)
        labels = np.array(self.all_labels)
        features = features.reshape((*features.shape, 1))
        return features, labels

buffer_size = 30
circular_buffer = deque(maxlen=buffer_size)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 3)  # Assuming 3 classes for gesture prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel()
model.load_state_dict(torch.load("pretrained.pth"))
model.eval()

def process_data(message):
    # Append data to the circular buffer
    circular_buffer.append(message)
    
    # Check if the buffer is full
    if len(circular_buffer) == buffer_size:
        # Convert the buffer to a numpy array
        buffer_data = np.array(list(circular_buffer))
        return buffer_data
    else:
        return None


async def handle_websocket(websocket, path):
    async for message in websocket:
        buffer_data = process_data(message)
        if buffer_data is not None:
            data = []
            for s in buffer_data:
                curr_list = s.split(',')
                new_list = []
                for char in curr_list:
                    if len(char) > 0:
                        new_list.append(float(char))
                new_list = np.array(new_list)
                data.append(new_list)
            data = np.array(data)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            scaled_data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            prediction = model(scaled_data_tensor)
            print(prediction)
        # print("Received message:", message)

start_server = websockets.serve(handle_websocket, 'localhost', 8000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
