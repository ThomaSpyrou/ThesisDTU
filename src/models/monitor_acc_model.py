import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load data from CSV file
df = pd.read_csv('final.csv')

# Define custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.inputs = data.iloc[:, 0].values
        self.targets = data.iloc[:, 1:].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y_true = torch.tensor(self.targets[idx][0], dtype=torch.float32)
        y_pred = torch.tensor(self.targets[idx][1], dtype=torch.long)
        return x, y_true, y_pred

# Instantiate dataset and dataloader
dataset = MyDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
class IamWatchingU(nn.Module):
    def __init__(self):
        super(IamWatchingU, self).__init__()
        self.l1 = nn.Linear(1, 100)
        self.l2 = nn.Linear(100, 200)
        self.l3 = nn.Linear(200, 200)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

model = IamWatchingU()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss() # to be changed

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (inputs, targets_true, targets_pred) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs.squeeze(), targets_true)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, loss: {running_loss/(batch_idx+1):.4f}')
