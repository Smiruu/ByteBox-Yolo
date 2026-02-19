import os

# Path to your rice dataset
base_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\dataset\white_rice"

# Subfolders containing labels
folders = ["train", "valid", "test"]

# Loop through each folder
for folder in folders:
    label_dir = os.path.join(base_path, folder, "labels")
    
    if not os.path.exists(label_dir):
        print(f"⚠ Skipping {label_dir}, folder not found.")
        continue

    for file_name in os.listdir(label_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(label_dir, file_name)
            
            # Read label file
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    parts[0] = '0'  # Set class ID to 0 for rice
                new_lines.append(" ".join(parts))
            
            # Overwrite with updated labels
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))
    
    print(f"✅ Reset class IDs to 0 in {label_dir}")

print("🎯 Rice labels reset successfully!")
