import os

# Path to your caldareta dataset
base_path = r"C:\Users\lanz\Desktop\ByteBox-Yolo\dataset\caldareta"

# Subfolders containing labels
folders = ["train", "valid", "test"]

for folder in folders:
    label_dir = os.path.join(base_path, folder, "labels")
    
    if not os.path.exists(label_dir):
        print(f"⚠ Skipping {label_dir}, folder not found.")
        continue

    for file_name in os.listdir(label_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(label_dir, file_name)
            
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    parts[0] = "3"  # ✅ Set class ID to 3
                new_lines.append(" ".join(parts))
            
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))
    
    print(f"✅ All labels in {label_dir} now start with 3")

print("🎯 Finished updating all Caldareta labels to class ID 3!")
