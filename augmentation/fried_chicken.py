import os
import cv2
import albumentations as A
from tqdm import tqdm

# Path to fried chicken dataset
base_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\dataset\fried_chicken"

# Augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=40, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=40, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
])

# Loop over dataset splits
for folder in ["train", "valid", "test"]:
    img_dir = os.path.join(base_path, folder, "images")
    if not os.path.exists(img_dir):
        print(f"⚠️ Folder does not exist: {img_dir}, skipping...")
        continue

    for img_name in tqdm(os.listdir(img_dir), desc=f"Augmenting images in {folder}"):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Failed to load image: {img_path}")
            continue

        # Create 3 augmented copies
        for i in range(3):
            augmented = transform(image=image)
            aug_img = augmented['image']

            # Save augmented image
            ext = os.path.splitext(img_name)[1]
            base = os.path.splitext(img_name)[0]
            aug_img_name = f"{base}_aug{i}{ext}"
            cv2.imwrite(os.path.join(img_dir, aug_img_name), aug_img)

print("✅ Fried chicken image augmentation completed!")
