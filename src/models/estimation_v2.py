import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.pred = data[:,0]
        self.true = data[:,1]
        self.pred_soft = data[:,2]
        self.true_soft = data[:,3]
        self.soft = data[:,4]

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, idx):
        sample = {'pred': self.pred[idx], 
                  'true': self.true[idx], 
                  'pred_soft': self.pred_soft[idx], 
                  'true_soft': self.true_soft[idx], 
                  'soft': self.soft[idx]}
        return sample

class IamWatchingU(nn.Module):
    def __init__(self):
        super(IamWatchingU, self).__init__()
        self.l1 = nn.Linear(100,150)
        self.l2 = nn.Linear(150, 150)
        self.l3 = nn.Linear(150, 100)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(100, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data['soft'].to(device), data['true_soft'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    return train_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            inputs, targets = data['soft'].to(device), data['true_soft'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            running_loss += loss.item()
    val_loss = running_loss / len(val_loader)
    return val_loss

def train_model(train_data, val_data, model, batch_size, epochs, lr, device):
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print('Epoch {}/{} | Train Loss: {:.6f} | Val Loss: {:.6f}')
