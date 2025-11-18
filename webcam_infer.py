# webcam_infer.py
# Usage:
#   .venv activated
#   python webcam_infer.py --checkpoint ./checkpoints/mobile_hybrid_best.pth --data_root ./data/kept_arith_ops
#
# Press:
#   c : capture current prediction (3 captures in sequence -> compute)
#   s : save debug frame
#   q : quit

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image

# -------------------------
# Args
# -------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--data_root", required=True, help="Path to dataset root (contains train/ or val/ with classes)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flip", action="store_true", help="Horizontally flip the webcam feed (mirror)")
    p.add_argument("--smoothing", type=int, default=5, help="How many frames to average probs over (temporal smoothing)")
    p.add_argument("--confidence", type=float, default=0.6, help="Min confidence to accept a displayed prediction")
    p.add_argument("--stability", type=int, default=3, help="How many of the recent frames must agree on the top label to accept a capture")
    p.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture()")
    return p.parse_args()

# -------------------------
# Build model (must match training architecture)
# -------------------------
def build_model(num_classes, device, pretrained=False):
    try:
        # prefer explicit weights API when available; but load without weights for inference
        model = models.mobilenet_v2(weights=None)
    except Exception:
        model = models.mobilenet_v2(pretrained=False)
    in_f = model.classifier[1].in_features if hasattr(model, "classifier") else model.last_channel
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    model.to(device)
    model.eval()
    return model

# -------------------------
# Transforms
# -------------------------
def build_val_tf(img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf

# -------------------------
# Result calc helper
# -------------------------
OP_MAP = {
    "A": "+",
    "S": "-",
    "X": "*",
    "D": "/",
    "P": "**"
}
def compute_result(a, op_sym, b):
    try:
        if op_sym == "/":
            # safe division: avoid ZeroDivisionError
            if float(b) == 0.0:
                return "Error: div by 0"
            return str(float(a) / float(b))
        elif op_sym == "**":
            return str(float(a) ** float(b))
        else:
            # + - *
            return str(eval(f"{float(a)} {op_sym} {float(b)}"))
    except Exception as e:
        return f"Err:{e}"

# -------------------------
# Main webcam loop
# -------------------------
def main():
    args = get_args()
    device = torch.device(args.device)

    # load dataset classes from train folder (so ordering matches checkpoint)
    train_dir = Path(args.data_root) / "train"
    if not train_dir.exists():
        raise SystemExit(f"train folder not found at: {train_dir} (required to infer class ordering)")

    # Use ImageFolder only to load class names (no heavy operations)
    dummy_tf = build_val_tf(args.img_size)
    ds = datasets.ImageFolder(str(train_dir), transform=dummy_tf)
    classes = ds.classes
    print("Classes:", classes)

    # Build model and load weights
    model = build_model(len(classes), device=device, pretrained=False)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck.get("model_state", ck)
    model.load_state_dict(state)
    model.eval()
    val_tf = build_val_tf(args.img_size)

    # smoothing buffers
    smoothing = max(1, args.smoothing)
    prob_buffers = [deque(maxlen=smoothing) for _ in range(len(classes))]  # not necessary to keep per-class, but keep for clarity
    history_prob = deque(maxlen=smoothing)  # we store probs vectors here

    # capture sequence storage
    capture_slots = []  # will store tuples (label, confidence, raw_value) e.g. ('3',0.92,'3')
    capture_text = ["A (operand)", "Operator", "B (operand)"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Check index or permissions.")

    print("Camera opened. Press 'q' to quit, 'c' to capture (A -> op -> B), 's' to save frame.")

    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed; exiting")
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        # copy for display & optional saving
        display = frame.copy()

        # prepare central crop / ROI if you want (here we use full frame resized)
        # convert BGR->RGB, to PIL, apply transform
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = val_tf(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]  # (num_classes,)

        # smoothing
        history_prob.append(probs)
        avg_prob = np.mean(np.stack(list(history_prob), axis=0), axis=0)
        pred_idx = int(np.argmax(avg_prob))
        pred_label = classes[pred_idx]
        pred_conf = float(avg_prob[pred_idx])

        # overlay prediction if confident
        if pred_conf >= args.confidence:
            text = f"Pred: {pred_label} ({pred_conf*100:.1f}%)"
            color = (0, 200, 0)
        else:
            text = f"Pred: --  ({pred_conf*100:.1f}%)"
            color = (0, 150, 255)

        # show top-3 probs for debugging
        top3 = np.argsort(avg_prob)[::-1][:3]
        top_text = " | ".join([f"{classes[i]}:{avg_prob[i]*100:.1f}%" for i in top3])

        # draw on image
        cv2.rectangle(display, (0,0), (display.shape[1], 36), (0,0,0), -1)
        cv2.putText(display, text, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(display, top_text, (8, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

        # show capture progress
        for i in range(len(capture_text)):
            y = 50 + i*26
            if i < len(capture_slots):
                st = f"[{i+1}] {capture_text[i]} => {capture_slots[i][0]} ({capture_slots[i][1]*100:.1f}%)"
                cv2.putText(display, st, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 1, cv2.LINE_AA)
            else:
                st = f"[{i+1}] {capture_text[i]} => (press 'c')"
                cv2.putText(display, st, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

        cv2.imshow("Gesture Calculator", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # save debug frame
            fname = f"webcam_debug_{int(time.time())}_{saved_idx}.jpg"
            cv2.imwrite(fname, frame)
            saved_idx += 1
            print("Saved", fname)
        elif key == ord("c"):
            # register current top prediction into capture slots, but require stability
            if len(history_prob) == 0:
                print("No history yet; try again in a moment.")
                continue

            # count how many recent frames agree on the current top label
            tops = [int(np.argmax(p)) for p in history_prob]
            top_agree = sum(1 for t in tops if t == pred_idx)
            if top_agree < max(1, args.stability):
                print(f"Prediction not stable: {top_agree}/{len(history_prob)} frames agree; need {args.stability}.")
                continue

            if pred_conf >= args.confidence:
                capture_slots.append((pred_label, pred_conf))
                print(f"Captured slot {len(capture_slots)}:", pred_label, pred_conf)
            else:
                print("Not confident enough to capture (move hand or lower confidence threshold).")

            # if we have 3 captures: compute result
            if len(capture_slots) == 3:
                op_label = capture_slots[1][0]
                op_sym = OP_MAP.get(op_label, None)
                a_label = capture_slots[0][0]
                b_label = capture_slots[2][0]

                # try to interpret operands as numbers
                if op_sym is None:
                    res = "Unknown operator"
                else:
                    try:
                        res = compute_result(float(a_label), op_sym, float(b_label))
                    except Exception as e:
                        res = f"Compute error: {e}"

                print("=== CALC ===")
                print(f"A: {a_label}   OP: {op_label} ({op_sym})   B: {b_label}")
                print("Result:", res)
                # overlay result briefly on screen (block loop for a second)
                overlay = display.copy()
                cv2.rectangle(overlay, (0, int(display.shape[0]/2)-40), (display.shape[1], int(display.shape[0]/2)+40), (0,0,0), -1)
                cv2.putText(overlay, f"{a_label} {op_sym} {b_label} = {res}", (20, int(display.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("Gesture Calculator", overlay)
                cv2.waitKey(1000)  # show for 1 second
                capture_slots = []  # reset

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
