import pandas as pd
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import ast
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['soft'] = self.data['soft'].apply(lambda x: ast.literal_eval(x))
        # self.data['label'] = self.data['label'].apply(lambda x: ast.literal_eval(x))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        soft_list = torch.tensor((self.data.iloc[idx]['soft']))
        true_soft = torch.tensor(self.data.iloc[idx]['true_soft'])

        return soft_list, true_soft

data_path_train = 'final100.csv'
# data_path_validation = 'cifar100validation.csv'
dataset_train = MyDataset(data_path_train)
# train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True) # 256, 512

# dataset_validation = MyDataset(data_path_validation)
# valid_data_loader = DataLoader(dataset_validation, batch_size=32, shuffle=True) # 256, 512


batch_size = 32
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset_train)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset_train, batch_size=16,
                                                sampler=valid_sampler)


class IamWatchingU(nn.Module):
    def __init__(self):
        super(IamWatchingU, self).__init__()
        self.l1 = nn.Linear(100,150)
        self.l2 = nn.Linear(150, 150)
        self.l3 = nn.Linear(150, 100)
        # self.l4 = nn.Linear(80, 40)
        # self.l5 = nn.Linear(800, 400)
        # self.l6 = nn.Linear(400, 200)
        # self.l7 = nn.Linear(200, 100)
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
        #x = self.l4(x)
        #x = self.activation(x)
        # x = self.l5(x)
        # x = self.activation(x)
        # x = self.l6(x)
        # x = self.activation(x)
        # x = self.l7(x)
        # x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

model = IamWatchingU()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()
threshold_acc = 0.5


def train_model(model, dataloader, optimizer, criterion, num_epochs=50):
    print("Start training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_to_plot = []
    val_loss_to_plot = []
    acc_total = []

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    
    for epoch in range(num_epochs):
        print("starting epoch: ", epoch)
        total_loss = 0.0
        correct = 0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            soft_max = torch.nn.functional.softmax(outputs, dim=1)
            out_tensor = soft_max.squeeze()

            mask =  out_tensor >= threshold_acc
            correct = mask.sum().item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, acc: {correct / len(dataloader):.4f}")
         
        loss_to_plot.append(total_loss/len(dataloader))
        acc_total.append(correct / len(dataloader))
        val_loss_to_plot.append(evaluate_model(model, validation_loader, criterion))


    plt.plot(loss_to_plot, label='Training loss')
    plt.plot(val_loss_to_plot, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("estimantion_cifar10_clean.png")

    
def evaluate_model(model, dataloader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            total_loss += loss.item()
            out_tensor = outputs.squeeze()

            mask =  out_tensor > 0.5
            correct = mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    print(f"val acc: {correct / len(dataloader):.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss


train_model(model, train_loader, optimizer, criterion)