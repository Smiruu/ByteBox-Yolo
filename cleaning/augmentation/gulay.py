import os
import cv2
import random
import shutil
import albumentations as A
from tqdm import tqdm

# ===== SETTINGS =====
original_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\clean_images\Gulay"
output_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\augmented_dataset\Gulay"
TARGET_TOTAL = 800
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

# Get valid images
image_files = [
    f for f in os.listdir(original_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(image_files) == 0:
    print("❌ No images found.")
    exit()

print("📦 Copying original images...")

# 1️⃣ Copy original 150 images
for img_name in image_files:
    src = os.path.join(original_path, img_name)
    dst = os.path.join(output_path, img_name)
    shutil.copy(src, dst)

current_total = len(os.listdir(output_path))
print(f"Original images copied: {current_total}")

# 2️⃣ Generate remaining augmentations
needed_aug = TARGET_TOTAL - current_total
print(f"🔄 Generating {needed_aug} augmented images...")

for i in tqdm(range(needed_aug), desc="Augmenting"):
    img_name = random.choice(image_files)
    img_path = os.path.join(original_path, img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    augmented = transform(image=image)
    aug_img = augmented["image"]

    base, ext = os.path.splitext(img_name)
    new_name = f"ginataang_gulay_aug_{i}{ext}"

    cv2.imwrite(os.path.join(output_path, new_name), aug_img)

print("✅ Done!")
print(f"Final total images: {len(os.listdir(output_path))}")
