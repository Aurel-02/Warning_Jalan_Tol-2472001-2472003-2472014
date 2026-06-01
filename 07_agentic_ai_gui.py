import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import threading
import pyttsx3
import time
import os

# --- Arsitektur Model CNN ---
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

# --- Sistem Agentic AI ---
class AgenticAI:
    def __init__(self):
        # Inisialisasi engine Text-to-Speech
        self.engine = pyttsx3.init()
        # Atur kecepatan bicara (words per minute)
        self.engine.setProperty('rate', 150)
        # Ambil suara bahasa indonesia jika tersedia, atau bahasa inggris secara default
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id) 
        
        self.is_alarming = False

    def play_alarm(self):
        self.is_alarming = True
        def alarm_thread():
            while self.is_alarming:
                # Menggunakan engine pyttsx3 pada thread terpisah
                # karena runAndWait() memblokir eksekusi
                temp_engine = pyttsx3.init()
                temp_engine.setProperty('rate', 150)
                temp_engine.say("Peringatan! Anda mengantuk! Segera bangun!")
                temp_engine.runAndWait()
                time.sleep(0.5)
                
        # Jalankan di daemon thread agar tidak memblokir GUI Tkinter
        threading.Thread(target=alarm_thread, daemon=True).start()

    def stop_alarm(self):
        self.is_alarming = False

    def find_nearest_rest_area(self):
        # Simulasi Pintar: Fungsi ini bertindak seperti Tool Call LLM
        # Dalam praktiknya, ini akan menembak Google Maps API / OpenStreetMap API
        return (
            "Agent menemukan bahwa lokasi Rest Area terdekat adalah:\n\n"
            "📍 Rest Area KM 57 (Jarak: 3 KM di depan)\n"
            "☕ Fasilitas: Toilet, Pom Bensin, Minimarket\n\n"
            "Agen merekomendasikan Anda untuk segera menepi dan beristirahat minimal 15 menit."
        )

# --- Aplikasi GUI Utama ---
class DrowsinessApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("700x650")
        self.window.configure(bg="#2c3e50")
        
        # 1. Load Model PyTorch
        self.model = TollRoadCNN()
        # Ambil lokasi absolut script ini
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'toll_road_cnn.pth')
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            print("Model berhasil dimuat!")
        else:
            messagebox.showerror("Error", f"File model {model_path} tidak ditemukan!")
            self.window.destroy()
            return

        # 2. Inisialisasi Agentic AI
        self.agent = AgenticAI()
        
        # 3. Buka Kamera (Webcam Default = 0)
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Kamera tidak terdeteksi!")
            self.window.destroy()
            return

        # Frame untuk video
        video_frame = tk.Frame(window, bg="#2c3e50")
        video_frame.pack(pady=10)

        # Canvas tempat menempelkan frame kamera
        width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = tk.Canvas(video_frame, width=width, height=height, bg="black")
        self.canvas.pack()

        # Label Status
        self.status_label = tk.Label(window, text="Status: NORMAL (Mata Terbuka)", 
                                     font=("Helvetica", 16, "bold"), fg="#27ae60", bg="#2c3e50")
        self.status_label.pack(pady=10)

        # Tombol untuk mematikan suara & cari rest area (Disembunyikan pada awalnya)
        self.btn_stop = tk.Button(window, text="SAYA SUDAH BANGUN (Cari Rest Area)", 
                                  command=self.on_stop_alarm, 
                                  bg="#e74c3c", fg="white", 
                                  font=("Helvetica", 14, "bold"),
                                  padx=20, pady=10, relief=tk.RAISED, cursor="hand2")
        self.btn_stop.pack(pady=10)
        self.btn_stop.pack_forget() # Sembunyikan tombol saat status Normal

        # Variabel Logika Deteksi
        self.closed_frames_count = 0
        # Toleransi: Misalnya jika 5 frame berturut-turut terdeteksi tertutup, anggap mengantuk
        self.drowsy_threshold = 5 
        self.is_drowsy_state = False

        # Memulai loop kamera
        self.delay = 15 # milidetik
        self.update()
        
        # Tangani ketika window ditutup silang
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
        
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Preprocessing frame dari kamera ke bentuk yang dipahami model
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Model dilatih dengan ukuran 64x64
            img_resized = cv2.resize(gray, (64, 64))
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Format PyTorch (batch_size=1, channel=1, H=64, W=64)
            img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)
            
            # Prediksi menggunakan model CNN
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, dim=0)
            
            # labels_map = {0: 'open', 1: 'closed'}
            pred_idx = predicted_class.item()
            
            # Update logika
            if pred_idx == 1: # Closed
                self.closed_frames_count += 1
                box_color = (0, 0, 255) # Merah BGR
                status_text = f"Mata Tertutup ({confidence.item()*100:.1f}%)"
            else: # Open
                self.closed_frames_count = max(0, self.closed_frames_count - 1)
                box_color = (0, 255, 0) # Hijau BGR
                status_text = f"Mata Terbuka ({confidence.item()*100:.1f}%)"

            # Picu Agen jika ambang batas terlampaui dan belum dalam status mengantuk
            if self.closed_frames_count >= self.drowsy_threshold and not self.is_drowsy_state:
                self.trigger_drowsy_alarm()
            
            # Gambar teks dan kotak di frame video
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)
            
            # Konversi kembali dari BGR (OpenCV) ke RGB (Tkinter/Pillow)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
        # Panggil fungsi ini lagi setelah {self.delay} milidetik
        self.window.after(self.delay, self.update)

    def trigger_drowsy_alarm(self):
        """Dipanggil saat model yakin pengemudi mengantuk."""
        self.is_drowsy_state = True
        self.status_label.config(text="Status: MENGANTUK! BAHAYA!", fg="#e74c3c") # Merah
        
        # Munculkan tombol interaktif
        self.btn_stop.pack(pady=10) 
        
        # Panggil agen untuk berteriak (suara)
        self.agent.play_alarm()

    def on_stop_alarm(self):
        """Dipanggil saat pengemudi menekan tombol 'Saya Sudah Bangun'."""
        # 1. Hentikan suara
        self.agent.stop_alarm()
        self.is_drowsy_state = False
        self.closed_frames_count = 0
        
        # 2. Kembalikan UI ke normal
        self.status_label.config(text="Status: NORMAL (Mata Terbuka)", fg="#27ae60")
        self.btn_stop.pack_forget() # Sembunyikan lagi tombolnya
        
        # 3. Agentic AI Action: Cari Rest Area dan berikan info
        rest_area_info = self.agent.find_nearest_rest_area()
        
        # Tampilkan hasil pencarian Agen dalam popup message
        messagebox.showinfo("Agentic AI - Rest Area Recommender", rest_area_info)

    def on_closing(self):
        """Dipanggil ketika pengguna menutup jendela aplikasi."""
        self.agent.stop_alarm()
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root, "Warning Jalan Tol - Agentic AI Integration")
