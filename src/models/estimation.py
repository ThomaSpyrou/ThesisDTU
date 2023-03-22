import pandas as pd
import torch
import numpy as np
import ast
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['soft_list'] = self.data['soft_list'].apply(lambda x: ast.literal_eval(x))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        soft_list = torch.tensor((self.data.iloc[idx]['soft_list']))
        pred_soft = torch.tensor(self.data.iloc[idx]['pred_soft'])

        return soft_list, pred_soft

data_path = 'final_1.csv'
dataset = MyDataset(data_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class IamWatchingU(nn.Module):
    def __init__(self):
        super(IamWatchingU, self).__init__()
        self.l1 = nn.Linear(10, 100)
        self.l2 = nn.Linear(100, 200)
        # self.l3 = nn.Linear(200, 400)
        # self.l4 = nn.Linear(400, 200)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        # x = self.l3(x)
        # x = self.relu(x)
        # x = self.l4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

model = IamWatchingU()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
criterion = nn.MSELoss() 

def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        

train_model(model, dataloader, optimizer, criterion)