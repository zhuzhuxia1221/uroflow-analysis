import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
best_model_dir = 'bestmodelcnn'
if not os.path.exists(best_model_dir):
    os.makedirs(best_model_dir)

x_data = pd.read_csv('input_x219.csv', header=None).values
y_data = pd.read_csv('input_y219.csv', header=None).values
labels = np.argmax(y_data, axis=1)  

data = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)  
labels = torch.tensor(labels, dtype=torch.int64)

train_ratio, val_ratio = 0.6, 0.2
train_size = int(train_ratio * len(data))
val_size = int(val_ratio * len(data))
test_size = len(data) - train_size - val_size

train_data, val_data, test_data = random_split(TensorDataset(data, labels), [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 7, padding=1)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=1)
        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.fc = nn.Linear(6016, 4)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return F.softmax(self.fc(x), dim=1)

model = Simple1DCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

