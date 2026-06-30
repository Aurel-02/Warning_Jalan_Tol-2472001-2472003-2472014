import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import os

# definisikan ulang arsitektur model cnn
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

# load model yang sudah terlatih
model = TollRoadCNN()
model.load_state_dict(torch.load('toll_road_cnn.pth'))
model.eval()

# load dataset mrl untuk mengambil sampel gambar secara acak
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "dataset", "dataset_mrl.csv")
df = pd.read_csv(csv_path)

# perbaiki path lokal
old_base = r"E:\sem 4\kecemes\dataset"
new_base = os.path.join(script_dir, "dataset")
df['image_path'] = df['image_path'].apply(lambda x: new_base + x[len(old_base):] if x.lower().startswith(old_base.lower()) else x)

# ambil satu contoh gambar acak dari dataset yang benar-benar ada di disk
sample = df.sample(1).iloc[0]
image_path = sample['image_path']
true_label = sample['label']

print(f"Menguji gambar sampel: {os.path.basename(image_path)}")
print(f"Label asli (Ground Truth): {true_label}")

# baca gambar dalam mode grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Gagal membaca gambar.")
else:
    # preprocess gambar (sesuai cara training)
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized.astype('float32') / 255.0
    
    # ubah shape menjadi format PyTorch (batch_size=1, channel=1, H=64, W=64)
    img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)
    
    # prediksi
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, dim=0)
        
    labels_map = {0: 'open', 1: 'closed'}
    predicted_label = labels_map[predicted_class.item()]
    
    print(f"Prediksi Model: {predicted_label} (Akurasi Keyakinan: {confidence.item() * 100:.2f}%)")
