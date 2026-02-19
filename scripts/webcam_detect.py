import cv2
from ultralytics import YOLO
from datetime import datetime
import numpy as np

# Load the trained YOLO model
model = YOLO(r"C:\Users\lanz\Desktop\ByteBox-Yolo\runs\detect\train\weights\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Settings
CONFIDENCE_THRESHOLD = 0.70
WHITE_THRESHOLD = 245
WHITE_PERCENT = 0.90

# 🔥 Automatically detect ALL model classes
TARGET_FOODS = {
    name.lower().replace(" ", "_")
    for name in model.names.values()
}

# Track foods currently logged
logged_foods = set()

print("🟢 YOLO detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame.")
        continue

    # Skip mostly white frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    white_ratio = np.mean(gray > WHITE_THRESHOLD)
    if white_ratio >= WHITE_PERCENT:
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Run YOLO detection
    results = model.predict(source=frame, conf=0.3, verbose=False)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)

    current_detected = set()

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls].lower().replace(" ", "_")

            if conf >= CONFIDENCE_THRESHOLD and label in TARGET_FOODS:
                current_detected.add(label)

    new_foods_to_log = current_detected - logged_foods

    if new_foods_to_log:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for label in new_foods_to_log:
            print(f"✅ {timestamp} - Detected: {label}")

        logged_foods.update(new_foods_to_log)

    # Remove foods that disappear
    logged_foods -= (logged_foods - current_detected)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
