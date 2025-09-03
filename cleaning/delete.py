import os

# Path to the dataset folder
dataset_path = "C:\\Users\\lanz\\Desktop\\ByteBox-Yolo\\dataset\\fried_chicken"

# Subfolders to check
splits = ["train", "test", "valid"]

for split in splits:
    images_folder = os.path.join(dataset_path, split, "images")
    labels_folder = os.path.join(dataset_path, split, "labels")
    
    if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
        continue
    
    # Get all image filenames without extension
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    
    # Loop through label files
    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):
            label_name = os.path.splitext(label_file)[0]
            if label_name not in image_files:
                label_path = os.path.join(labels_folder, label_file)
                os.remove(label_path)
                print(f"Deleted {label_file} from {split}/labels")

print("Cleanup complete!")
