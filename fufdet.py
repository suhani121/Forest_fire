import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
import time

# -------------------------------
# Coordinate Attention (CA)
# -------------------------------
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        reduced_channels = max(8, in_channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

# -------------------------------
# FuF-Det Classifier
# -------------------------------
class FuFDetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.ca = CoordAttention(512)  # ResNet18 has 512 channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.ca(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return torch.sigmoid(x)

# -------------------------------
# Training + Validation
# -------------------------------
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_path = 'forest fire'  # Replace with your dataset path
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = FuFDetClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(10):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1:2} | Train Acc: {train_acc:.4f}, Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}, Loss: {avg_val_loss:.4f} | Time: {time.time() - start_time:.2f}s")

    torch.save(model.state_dict(), "fufdet_classifier.pth")



