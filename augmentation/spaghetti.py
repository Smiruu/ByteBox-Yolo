import os
import cv2
import shutil
import albumentations as A
from tqdm import tqdm

# Paths
original_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\clean_images\Spaghetti"
output_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\augmented_dataset\Spaghetti"

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=40, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=40, p=0.6, border_mode=cv2.BORDER_CONSTANT),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
])

# Iterate through original images
for img_name in tqdm(os.listdir(original_path), desc="Copying & Augmenting Spaghetti Images"):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(original_path, img_name)

    # Copy original image to output folder
    shutil.copy(img_path, os.path.join(output_path, img_name))

    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Failed to load image: {img_path}")
        continue

    # Create 3 augmented copies per image
    for i in range(3):
        augmented = transform(image=image)
        aug_img = augmented['image']

        ext = os.path.splitext(img_name)[1]
        base = os.path.splitext(img_name)[0]
        aug_img_name = f"{base}_aug{i}{ext}"

        cv2.imwrite(os.path.join(output_path, aug_img_name), aug_img)

print("✅ Spaghetti image augmentation completed!")
print(f"All images (original + augmented) saved to: {output_path}")
