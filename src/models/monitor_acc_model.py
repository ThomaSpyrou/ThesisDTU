import pandas as pd
import torch
import ast
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


df = pd.read_csv('final_1.csv')
df['soft_list'] = df['soft_list'].apply(lambda x: ast.literal_eval(x))


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

dataset = MyDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class IamWatchingU(nn.Module):
    def __init__(self):
        super(IamWatchingU, self).__init__()
        self.l1 = nn.Linear(10, 100)
        self.l2 = nn.Linear(100, 200)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

model = IamWatchingU()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss() 

num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (inputs, targets_true, targets_pred) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs.squeeze(), targets_true)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predicted_values = outputs.detach().numpy()

        
    print(f'Epoch {epoch+1}, loss: {running_loss/(batch_idx+1):.4f}')
