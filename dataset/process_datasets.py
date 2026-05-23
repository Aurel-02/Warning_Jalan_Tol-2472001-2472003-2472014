import os
import glob
import csv

BASE_DIR = r"E:\sem 4\kecemes\dataset"
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset_gabungan.csv")

def find_image_for_label(label_path):
    # e.g., .../labels/abc.txt -> .../images/abc.*
    base = os.path.splitext(label_path)[0]
    base = base.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep)
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
        if os.path.exists(base + ext):
            return base + ext
    return None

def process_mrl(records):
    print("Processing MRL Dataset...")
    mrl_dir = os.path.join(BASE_DIR, "mrl_dataset")
    if not os.path.exists(mrl_dir): return
    for root, dirs, files in os.walk(mrl_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, f)
                # Check folder name for label
                folder_name = os.path.basename(root).lower()
                if "close" in folder_name:
                    records.append((path, "closed", "mrl_dataset"))
                elif "open" in folder_name:
                    records.append((path, "open", "mrl_dataset"))

def process_drowsy(records):
    print("Processing Drowsy Dataset...")
    drowsy_dir = os.path.join(BASE_DIR, "Drowsy_datset")
    if not os.path.exists(drowsy_dir): return
    for root, dirs, files in os.walk(drowsy_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, f)
                folder_name = os.path.basename(root).lower()
                if "drowsy" in folder_name:
                    records.append((path, "closed", "drowsy_dataset"))
                elif "natural" in folder_name:
                    records.append((path, "open", "drowsy_dataset"))

def process_yolo1(records):
    print("Processing YOLO Dataset 1 (Root)...")
    # Classes: 0='Eyeclosed', 1='Eyeopen', 2='Yawn'
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(BASE_DIR, split, "labels")
        if not os.path.exists(labels_dir): continue
        for root, dirs, files in os.walk(labels_dir):
            for f in files:
                if f.endswith('.txt'):
                    label_path = os.path.join(root, f)
                    with open(label_path, 'r') as file:
                        lines = file.readlines()
                    
                    is_closed = False
                    is_open = False
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        if cls_id in [0, 2]: # Eyeclosed, Yawn
                            is_closed = True
                        elif cls_id == 1:
                            is_open = True
                    
                    label = None
                    if is_closed:
                        label = "closed"
                    elif is_open:
                        label = "open"
                        
                    if label:
                        img_path = find_image_for_label(label_path)
                        if img_path:
                            records.append((img_path, label, "yolo_dataset_1"))

def process_yolo2(records):
    print("Processing YOLO Dataset 2 (DMS)...")
    dms_dir = os.path.join(BASE_DIR, "DMS - Driver Monitoring System")
    if not os.path.exists(dms_dir): return
    # Classes: 'Closed Eye'=0, 'Normal'=3, 'sleepy'=9, 'yawns'=10
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dms_dir, split, "labels")
        if not os.path.exists(labels_dir): continue
        for root, dirs, files in os.walk(labels_dir):
            for f in files:
                if f.endswith('.txt'):
                    label_path = os.path.join(root, f)
                    with open(label_path, 'r') as file:
                        lines = file.readlines()
                    
                    is_closed = False
                    is_open = False
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        if cls_id in [0, 9, 10]: # Closed Eye, sleepy, yawns
                            is_closed = True
                        elif cls_id == 3: # Normal
                            is_open = True
                    
                    label = None
                    if is_closed:
                        label = "closed"
                    elif is_open:
                        label = "open"
                        
                    if label:
                        img_path = find_image_for_label(label_path)
                        if img_path:
                            records.append((img_path, label, "yolo_dataset_2 (dms)"))

def main():
    records = [] # (image_path, label, source)
    process_mrl(records)
    process_drowsy(records)
    process_yolo1(records)
    process_yolo2(records)
    
    print(f"Total valid records found: {len(records)}")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'source_dataset'])
        writer.writerows(records)
        
    print(f"Successfully saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
