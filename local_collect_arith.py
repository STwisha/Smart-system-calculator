# local_collect_arith.py
# Finds class folders (0-9 and A,S,X,D,P) anywhere inside the project
# and copies their files into data/kept_arith_ops/train/<label>.

import shutil
from pathlib import Path

PROJECT = Path.cwd()
DEST_BASE = PROJECT / "data" / "kept_arith_ops" / "train"
LABELS = [str(i) for i in range(10)] + ["A","S","X","D","P"]

print("Project:", PROJECT)
print("Destination base:", DEST_BASE)
DEST_BASE.mkdir(parents=True, exist_ok=True)

found_any = False
for lbl in LABELS:
    # search for folders named exactly lbl (case-sensitive)
    matches = list(PROJECT.rglob(lbl))
    # filter to directories only
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        print(f"[NOT FOUND] label folder '{lbl}' not found in project.")
        continue

    # use first match (if multiple)
    src_dir = matches[0]
    dest_dir = DEST_BASE / lbl
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in src_dir.iterdir() if p.is_file()]
    if not files:
        print(f"[EMPTY] source folder {src_dir} is empty for label '{lbl}'.")
        continue

    print(f"[COPY] {len(files)} files from {src_dir} -> {dest_dir}")
    for f in files:
        # copy each file if doesn't already exist in dest (prevents duplicates)
        dst = dest_dir / f.name
        if dst.exists():
            i = 1
            stem = f.stem
            suff = f.suffix
            while dst.exists():
                dst = dest_dir / f"{stem}_{i}{suff}"
                i += 1
        shutil.copy2(str(f), str(dst))
    found_any = True

if not found_any:
    print("\nNo label folders copied. Make sure your unzipped datasets are inside the project and contain folders named 0..9 and A,S,X,D,P.")
else:
    print("\nDone. Check the data/kept_arith_ops/train/ folder for copied files.")
