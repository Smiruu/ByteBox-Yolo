import os
import shutil

# ====== CONFIG ======
# List all dataset folders here
datasets = [
    r"C:\Users\lanz\Desktop\ByteBox-Yolo\dataset\white_rice",
    r"C:\Users\lanz\Desktop\ByteBox-Yolo\dataset\filipino_spaghetti",
    r"C:\Users\lanz\Desktop\ByteBox-Yolo\dataset\pancit_bihon",
    r"C:\Users\lanz\Desktop\ByteBox-Yolo\dataset\caldareta", 
    r"C:\Users\lanz\Desktop\ByteBox-Yolo\dataset\ginataang_gulay",


    # Add more datasets here later...
]

# Output merged dataset folder
output_dir = r"C:\Users\lanz\Desktop\ByteBox-Yolo\merged_dataset"
splits = ["train", "valid", "test"]

# ====================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_datasets():
    ensure_dir(output_dir)

    for split in splits:
        ensure_dir(os.path.join(output_dir, split, "images"))
        ensure_dir(os.path.join(output_dir, split, "labels"))

    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path.rstrip("\\/"))

        for split in splits:
            img_dir = os.path.join(dataset_path, split, "images")
            lbl_dir = os.path.join(dataset_path, split, "labels")

            if not os.path.exists(img_dir):
                continue

            for filename in os.listdir(img_dir):
                img_src = os.path.join(img_dir, filename)
                lbl_src = os.path.join(lbl_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

                if not os.path.exists(lbl_src):
                    continue

                # Rename with prefix to avoid conflicts
                new_img_name = f"{dataset_name}_{filename}"
                new_lbl_name = new_img_name.rsplit(".", 1)[0] + ".txt"

                img_dst = os.path.join(output_dir, split, "images", new_img_name)
                lbl_dst = os.path.join(output_dir, split, "labels", new_lbl_name)

                shutil.copy(img_src, img_dst)
                shutil.copy(lbl_src, lbl_dst)

    print(f"✅ Merge complete! Merged dataset saved to: {output_dir}")

if __name__ == "__main__":
    merge_datasets()
