# KEEP-ONLY-ARITHMETIC-SIGNS.py
# Run this from an environment that has access to the extracted dataset path.
# This script moves the chosen labels to a new folder. To permanently delete
# the others, see the commented "delete" lines below.

import os
import shutil
from pathlib import Path

# ---- EDIT THIS if your dataset path differs ----
dataset_base = Path("/mnt/data/extracted_dataset/mini_asl_alphabet")
train_folder = dataset_base / "asl_alphabet_train"
test_folder = dataset_base / "asl_alphabet_test"   # may exist, handle if not present

# Chosen mapping: operation -> label folder name in your dataset
chosen_labels = {
    "add": "A",
    "subtract": "S",
    "multiply": "X",
    "divide": "D",
    "exponent": "P"
}

# Destination (safe) where the kept labels will be moved
dest_base = Path("/mnt/data/kept_arith_ops")
dest_train = dest_base / "train"
dest_test = dest_base / "test"

# Create destination folders
dest_train.mkdir(parents=True, exist_ok=True)
dest_test.mkdir(parents=True, exist_ok=True)

def keep_only(folder_path: Path, dest_folder: Path, chosen_set: set):
    """
    Move chosen label directories from folder_path to dest_folder.
    All other sibling directories will be left in place (or deleted if you
    choose to enable deletion).
    """
    if not folder_path.exists():
        print(f"Source folder does not exist: {folder_path}")
        return

    # iterate all subfolders (each subfolder is a label)
    for entry in sorted(folder_path.iterdir()):
        if not entry.is_dir():
            continue
        label = entry.name
        if label in chosen_set:
            target = dest_folder / label
            if target.exists():
                print(f"Target already exists, will remove it first: {target}")
                shutil.rmtree(target)
            print(f"Moving '{entry}' -> '{target}'")
            shutil.move(str(entry), str(target))
        else:
            # Option A (safe): keep them where they are (do nothing)
            print(f"Keeping (not chosen) label folder untouched: {entry}")

            # Option B (move to trash folder instead of deleting):
            # trash = dest_base / "removed_labels"
            # trash.mkdir(parents=True, exist_ok=True)
            # print(f"Moving (not chosen) '{entry}' -> '{trash / label}'")
            # shutil.move(str(entry), str(trash / label))

            # Option C (PERMANENT DELETE) -- UNCOMMENT TO ENABLE (IRREVERSIBLE):
            # print(f"Deleting (permanently) folder: {entry}")
            # shutil.rmtree(entry)

# run for train and test
chosen_set = set(chosen_labels.values())
keep_only(train_folder, dest_train, chosen_set)
keep_only(test_folder, dest_test, chosen_set)

print("\nDone. Kept labels:")
for op, lab in chosen_labels.items():
    print(f"  {op:9s} -> {lab}")

print("\nKept data root:", dest_base)
print("If you moved items you can inspect", dest_train, "and", dest_test)
