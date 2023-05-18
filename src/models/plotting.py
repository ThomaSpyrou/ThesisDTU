import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 50

# Define transforms for data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Define ResNet50 model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train model
train_losses = []
train_accs = []
test_losses = []
test_accs = []
f1_scores = []

for epoch in range(num_epochs):
    train_loss = 0.0
    train_total = 0
    train_correct = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_losses.append(train_loss / len(trainloader))
    train_acc = 100 * train_correct / train_total
    train_accs.append(train_acc)
    
    test_loss = 0.0
    test_total = 0
    test_correct = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            y_true += labels.tolist()
            y_pred += predicted.tolist()
    
    test_losses.append(test_loss / len(testloader))
    test_acc = 100 * test_correct / test_total
    test_accs.append(test_acc)
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_scores.append(f1)
    
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%, F1 Score: {:.4f}'.format(epoch+1, num_epochs, train_losses[-1], train_accs[-1], test_losses[-1], test_accs[-1], f1_scores[-1]))

# Visualize results
# Visualize results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(train_losses, label='Train Loss')
axs[0].plot(test_losses, label='Test Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(train_accs, label='Train Acc')
axs[1].plot(test_accs, label='Test Acc')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy (%)')
axs[1].legend()

axs[2].plot(f1_scores, label='F1 Score')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('F1 Score')
axs[2].legend()

# Save each plot in a different file
plt.savefig('train_loss.png')
plt.close()

fig, axs = plt.subplots(figsize=(10, 5))
axs.plot(train_accs, label='Train Acc')
axs.plot(test_accs, label='Test Acc')
axs.set_xlabel('Epochs')
axs.set_ylabel('Accuracy (%)')
axs.legend()

plt.savefig('accuracy.png')
plt.close()

fig, axs = plt.subplots(figsize=(10, 5))
axs.plot(f1_scores, label='F1 Score')
axs.set_xlabel('Epochs')
axs.set_ylabel('F1 Score')
axs.legend()

plt.savefig('f1_score.png')
plt.close()

