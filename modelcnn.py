import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# ---------------------------
# FireSmokeCNN Model
# ---------------------------
class FireSmokeCNN(nn.Module):
    def __init__(self):
        super(FireSmokeCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2a = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 22 * 30, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.sigmoid(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x1 = F.sigmoid(self.conv2a(x))
        x2 = F.sigmoid(self.conv2b(x))
        x = torch.cat((x1, x2), dim=1)
        x = self.pool2(self.bn2(x))

        x = self.bn3(F.sigmoid(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = self.dropout(F.sigmoid(self.fc1(x)))
        x = self.dropout(F.sigmoid(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

# ---------------------------
# Training + Validation
# ---------------------------
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((90, 120)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder('forest fire', transform=transform)

    # Split: 80% training, 20% validation
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FireSmokeCNN().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    model.train()
    for epoch in range(50):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        train_acc = 100 * correct_train / total_train

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100 * correct_val / total_val
        model.train()

        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), 'fire_smoke_cnn.pth')

