import os
import cv2
import random
import albumentations as A
from tqdm import tqdm

# ===== SETTINGS =====
original_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\clean_images\Caldareta"
output_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\augmented_dataset\Caldareta"
TARGET_TOTAL = 469
# ====================

os.makedirs(output_path, exist_ok=True)

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=40, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.2,
        rotate_limit=40,
        p=0.6,
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
])

image_files = [
    f for f in os.listdir(original_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(image_files) == 0:
    print("❌ No images found in clean dataset folder.")
    exit()

print("🔄 Generating 469 augmented Caldareta images...")

for i in tqdm(range(TARGET_TOTAL), desc="Augmenting Images"):
    img_name = random.choice(image_files)
    img_path = os.path.join(original_path, img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    augmented = transform(image=image)
    aug_img = augmented["image"]

    base, ext = os.path.splitext(img_name)
    new_name = f"caldareta_aug_{i}{ext}"

    cv2.imwrite(os.path.join(output_path, new_name), aug_img)

print("✅ Done!")
print(f"Total Caldareta images generated: {TARGET_TOTAL}")
