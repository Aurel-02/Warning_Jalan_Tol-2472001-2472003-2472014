import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# load data yang sudah diproses
X = np.load('X_cnn.npy')
y = np.load('y.npy')

# ubah shape agar sesuai format PyTorch (batch, channel, height, width)
X = X.transpose(0, 3, 1, 2)

# bagi dataset jadi train (80%), val (10%), dan test (10%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp)

# dataset & dataloader untuk training
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# arsitektur model cnn untuk klasifikasi
class TollRoadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        return self.fc(self.conv(x))

model = TollRoadCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
epochs = 10
print("Mulai training model...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    train_loss = total_loss / total
    train_acc = correct / total
    
    # validasi tiap epoch
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)
            
    val_loss /= val_total
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

# evaluasi model di data test
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        _, preds = outputs.max(1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

print("\nHasil Evaluasi Data Test:")
print(classification_report(y_true, y_pred, target_names=['Open (0)', 'Closed (1)']))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# simpan bobot model
torch.save(model.state_dict(), 'toll_road_cnn.pth')
print("\nModel berhasil disimpan ke 'toll_road_cnn.pth'")
