import torch
import torch.nn as nn

# definisikan ulang arsitektur model
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

# load model
model = TollRoadCNN()
model.load_state_dict(torch.load('toll_road_cnn.pth'))
model.eval()

print("=== Struktur Model ===")
print(model)

print("\n=== Ukuran Parameter (Weights/Biases) ===")
for name, param in model.state_dict().items():
    print(f"{name}: {param.shape}")
