import cv2
import os
import time
from ultralytics import YOLO
from datetime import datetime
import numpy as np

# Load the trained YOLOv8 model

# CHANGE THE MODEL PATH TO YOUR OWN PATH TO RUN THE SCRIPT 
model = YOLO(r"C:\Users\lanz\Desktop\ByteBox-Yolo\runs\detect\train12\weights\best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Folder to save detected images
save_dir = "detected_items"
os.makedirs(save_dir, exist_ok=True)

# Settings
last_detection_time = 0
DETECTION_INTERVAL = 300  # seconds (5 minutes)
CONFIDENCE_THRESHOLD = 0.70  # minimum confidence to save
WHITE_THRESHOLD = 245  # pixel value threshold to consider "white"
WHITE_PERCENT = 0.90  # if 90%+ of pixels are white, skip

print("🟢 YOLOv8 detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame.")
        continue

    # Check for mostly white frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white_ratio = np.mean(gray > WHITE_THRESHOLD)
    if white_ratio >= WHITE_PERCENT:
        print("⚪ Frame is mostly white. Skipping detection...")
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Run YOLO detection
    results = model.predict(source=frame, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    current_time = time.time()
    detected_items = set()

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            if conf >= CONFIDENCE_THRESHOLD and label in {"rice", "pandesal", "fried chicken"}:
                detected_items.add(label)

    if detected_items:
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            print(f"✅ Detected: {', '.join(detected_items)} with ≥{CONFIDENCE_THRESHOLD*100}% confidence! Saving image(s)...")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for label in detected_items:
                class_dir = os.path.join(save_dir, label)
                os.makedirs(class_dir, exist_ok=True)
                filename = f"{label}_{timestamp}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, annotated_frame)
            last_detection_time = current_time
        else:
            remaining = int(DETECTION_INTERVAL - (current_time - last_detection_time))
            print(f"⏳ Waiting {remaining}s before saving another image.")
    else:
        print("❌ No food detected with ≥70% confidence.")

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
