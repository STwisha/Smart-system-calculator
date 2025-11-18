# make_test_split.py
from pathlib import Path
import random, shutil

root = Path("data/kept_arith_ops")
train = root / "train"
test = root / "test"
test.mkdir(parents=True, exist_ok=True)

print("Creating test split under:", test)

for cls in sorted([p for p in train.iterdir() if p.is_dir()]):
    dest = test / cls.name
    dest.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in cls.glob("*.*") if p.is_file()]
    if not imgs:
        print(f"  [SKIP] {cls.name} has no files.")
        continue
    k = max(1, int(0.1 * len(imgs)))   # 10% -> at least 1 file
    chosen = random.sample(imgs, k)
    for p in chosen:
        shutil.move(str(p), str(dest / p.name))
    print(f"  Moved {len(chosen)} from {cls.name} -> test/{cls.name}")
print("Done.")
