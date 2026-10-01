# Databricks notebook source
# MAGIC %md
# MAGIC # Corrected testing: QLoRA, q_proj, r = 16
# MAGIC
# MAGIC Re-evaluates the saved best-validation checkpoint of **QLoRA, q_proj, r = 16** (original notebook: `Augmentation =1 in Validation Set/Q-lora Fewer Layers R = 16`)
# MAGIC on the 211,800-crop test set with the **same input scaling as in training and validation**.
# MAGIC
# MAGIC * **Training and validation:** the model wrapper inverted the ImageNet normalization before DINOv3 (`x_denorm = x * std + mean`, clamped to [0, 1]) and ran under fp16 autocast.
# MAGIC * **Original test evaluation (Module 11):** the ImageNet-normalized tensor was passed to the backbone directly (`model(pixel_values=x)`), i.e. a different input scaling from training.
# MAGIC * **This notebook:** evaluates the test set exactly as validation was evaluated during training: inverse normalization to [0, 1], fp16 autocast, and the 4-bit base prepared with `prepare_model_for_kbit_training` (non-quantized layers in fp32). Nothing is retrained; the adapters and head are the files saved at the best validation epoch in `dinov3_qlora_valx1`.
# MAGIC
# MAGIC Widget `preprocessing = legacy` reproduces the original evaluation (use with `limit = 2000` as a sanity check against the original predictions); `corrected` is the fixed protocol used for the paper.
# MAGIC Results are written to `dbfs:/mnt/playbehavior/Fine_tuning/Processed_Datasets/corrected_testing/dinov3_qlora_valx1/`.

# COMMAND ----------

# MAGIC %pip install --quiet transformers==4.57.1 accelerate==0.33.0 peft==0.13.2 bitsandbytes==0.48.0 albumentations==1.3.1 opencv-python-headless==4.8.1.78

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown("preprocessing", "corrected", ["corrected", "legacy"], "Input scaling at test time")
dbutils.widgets.text("limit", "0", "Evaluate only N crops (0 = all 211,800)")
dbutils.widgets.dropdown("subset", "random", ["random", "first"], "Which N crops (seeded random sample or first N)")
dbutils.widgets.text("batch_size", "16", "Batch size")
dbutils.widgets.text("num_workers", "4", "DataLoader workers")
dbutils.widgets.text("tag", "", "Optional tag for output file names")

# COMMAND ----------

import os, json, time, subprocess, datetime, socket
import numpy as np, pandas as pd, torch, torch.nn as nn, cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, Subset
from huggingface_hub import login
from transformers import AutoModel, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
import transformers, peft, bitsandbytes
from sklearn.metrics import classification_report, f1_score

MODEL_KEY = "Q_q16"
DISPLAY = "QLoRA, q_proj, r = 16"
RUN_DIR = "/dbfs/mnt/playbehavior/Fine_tuning/Processed_Datasets/dinov3_qlora_valx1"
ORIGINAL_PREDICTIONS = 'dinov3_qlora_predictions_valx1.csv'
ORIGINAL_ACCURACY = 0.7838
LEGACY_AUTOCAST = True        # how the original test cell ran (QLoRA notebooks: fp16 autocast; DoRA notebooks: none)
BASE_MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
DATA_DIR = "/dbfs/mnt/playbehavior/Fine_tuning/Processed_Datasets"
OUT_DIR = f"{DATA_DIR}/corrected_testing/dinov3_qlora_valx1"
LABELS = ["Drinking", "Feeding head down", "Feeding head up", "Lying", "Standing", "Walking", "frontal_pushing", "gallop", "leap"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

PREPROCESSING = dbutils.widgets.get("preprocessing")
LIMIT = int(dbutils.widgets.get("limit") or 0)
BATCH_SIZE = int(dbutils.widgets.get("batch_size") or 16)
NUM_WORKERS = int(dbutils.widgets.get("num_workers") or 4)
TAG = dbutils.widgets.get("tag").strip()
USE_AUTOCAST = True if PREPROCESSING == "corrected" else LEGACY_AUTOCAST
SUBSET = dbutils.widgets.get("subset")
SUFFIX = PREPROCESSING + (f"_{SUBSET}{LIMIT}" if LIMIT else "") + (f"_{TAG}" if TAG else "")
os.makedirs(OUT_DIR, exist_ok=True)
STARTED = datetime.datetime.utcnow().isoformat() + "Z"

def gpu_status():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"nvidia-smi unavailable ({e})"

print(f"Model: {DISPLAY} ({MODEL_KEY}) | preprocessing: {PREPROCESSING} | fp16 autocast: {USE_AUTOCAST} | limit: {LIMIT or 'all'}")
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} | bitsandbytes {bitsandbytes.__version__}")
print(f"host {socket.gethostname()} | GPU (name, MiB used, MiB total, util %): {gpu_status()}")
login(token=dbutils.secrets.get(scope="hf", key="token"))   # DINOv3 is a gated model

# COMMAND ----------

# Test set in the original order: all MmCows crops, then all PlayBehavior crops (as combined_test_df in Module 0)
mm = pd.read_csv(f"{DATA_DIR}/test_mmcows_dataset.csv")
pb = pd.read_csv(f"{DATA_DIR}/test_playbehavior_dataset.csv")
test_df = pd.concat([mm, pb], ignore_index=True)
assert len(test_df) == 211800, len(test_df)
test_df["test_index"] = np.arange(len(test_df))           # row number in the full combined test set
if LIMIT:
    sel = test_df.sample(n=LIMIT, random_state=0).sort_index() if SUBSET == "random" else test_df.iloc[:LIMIT]
    test_df = sel.reset_index(drop=True)
label_to_idx = {l: i for i, l in enumerate(LABELS)}
transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])   # = AugmentationPipeline(mode='test')

def to_local(p):
    return "/dbfs" + p[5:] if p.startswith("dbfs:/") else p

class TestSet(Dataset):
    def __len__(self):
        return len(test_df)
    def __getitem__(self, i):
        img = cv2.imread(to_local(test_df.at[i, "image_path"]))
        if img is None:
            raise FileNotFoundError(test_df.at[i, "image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return transform(image=img)["image"], label_to_idx[test_df.at[i, "label"]], i

# resume support: predictions are checkpointed every 30 min
CKPT = f"{OUT_DIR}/partial_{SUFFIX}.npz"
preds = np.full(len(test_df), -1, dtype=np.int64)
conf = np.zeros(len(test_df), dtype=np.float32)
if os.path.exists(CKPT):
    z = np.load(CKPT)
    preds[:] = z["preds"]; conf[:] = z["conf"]
    print(f"Resuming from checkpoint: {(preds >= 0).sum():,} crops already done")
todo = np.where(preds < 0)[0]
loader = DataLoader(Subset(TestSet(), todo.tolist()), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
print(f"Test crops: {len(test_df):,} | to evaluate: {len(todo):,} | batches: {len(loader):,}")

# COMMAND ----------

# Model exactly as trained: 4-bit NF4 base (double quantization, fp16 compute), prepared as in training, saved adapters and head
t_load = time.time()
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True)
base = AutoModel.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
if PREPROCESSING == "corrected":
    # as during training and validation: non-quantized layers cast to fp32 (the original test cells skipped this step)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
model = PeftModel.from_pretrained(base, f"{RUN_DIR}/best_lora_adapters").eval()
adapter_cfg = json.load(open(f"{RUN_DIR}/best_lora_adapters/adapter_config.json"))
print("Adapter config:", {k: adapter_cfg.get(k) for k in ("r", "lora_alpha", "lora_dropout", "use_dora", "target_modules")})
head = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
                     nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(512, 9))
head.load_state_dict(torch.load(f"{RUN_DIR}/best_classifier.pth", map_location="cpu"))
device = next(model.parameters()).device
head = head.to(device).eval()
mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
LOAD_SECONDS = time.time() - t_load
print(f"Model ready in {LOAD_SECONDS:.0f} s on {device}; allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB | GPU: {gpu_status()}")

# COMMAND ----------

torch.cuda.reset_peak_memory_stats()
t0 = last_log = last_ckpt = time.time()
done, gpu_log = 0, []
with torch.no_grad():
    for b, (x, y, ii) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        if PREPROCESSING == "corrected":
            x = torch.clamp(x * std + mean, 0, 1)            # identical to the training/validation wrapper
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AUTOCAST):
            feats = model(pixel_values=x).pooler_output.float()
            logits = head(feats)
        p = torch.softmax(logits.float(), dim=1)
        c, k = p.max(dim=1)
        idx = ii.numpy()
        preds[idx] = k.cpu().numpy(); conf[idx] = c.cpu().numpy()
        done += len(idx)
        now = time.time()
        if now - last_log > 600 or b == len(loader) - 1:
            rate = done / (now - t0)
            g = gpu_status(); gpu_log.append(g)
            print(f"[{datetime.datetime.utcnow():%H:%M} UTC] {done:,}/{len(todo):,} | {rate:.2f} crops/s | ETA {(len(todo) - done) / max(rate, 1e-9) / 3600:.1f} h"
                  f" | peak allocated {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | GPU {g}", flush=True)
            last_log = now
        if now - last_ckpt > 1800:
            np.savez(CKPT, preds=preds, conf=conf); last_ckpt = now
EVAL_SECONDS = time.time() - t0
assert (preds >= 0).all()

# COMMAND ----------

trues = test_df["label"].map(label_to_idx).values
cm = np.zeros((9, 9), dtype=np.int64)
np.add.at(cm, (trues, preds), 1)
acc = float(np.trace(cm) / cm.sum())
src = test_df["source"].values
per_source = {s: float((preds[src == s] == trues[src == s]).mean()) for s in ("mmcows", "playbehavior") if (src == s).any()}
print(f"Accuracy ({PREPROCESSING}): {acc:.4f} | original reported: {ORIGINAL_ACCURACY:.4f} | per source: {per_source}")
print(classification_report(trues, preds, labels=list(range(9)), target_names=LABELS, digits=4, zero_division=0))

comparison = None
if ORIGINAL_PREDICTIONS:
    o = pd.read_csv(f"{RUN_DIR}/{ORIGINAL_PREDICTIONS}")
    pred_col = "pred_label" if "pred_label" in o.columns else [c for c in o.columns if "pred" in c.lower()][0]
    o = o.iloc[test_df["test_index"].values]
    assert (o["image_path"].values == test_df["image_path"].values).all(), "order differs from the original predictions"
    o_pred = o[pred_col].map(label_to_idx).values
    comparison = {"agreement_with_original": float((o_pred == preds).mean()),
                  "original_accuracy_same_crops": float((o_pred == trues).mean()),
                  "changed_predictions": int((o_pred != preds).sum())}
    print("Comparison with the original predictions:", comparison)

result = {
    "key": MODEL_KEY, "display": DISPLAY, "run_dir": "dinov3_qlora_valx1",
    "preprocessing": "corrected: inverse ImageNet normalization to [0, 1] as in training" if PREPROCESSING == "corrected"
                     else "legacy (normalized input, as in the original evaluation)",
    "autocast_fp16": USE_AUTOCAST, "n": int(cm.sum()), "accuracy": acc, "per_source_accuracy": per_source,
    "macro_f1": float(f1_score(trues, preds, average="macro")), "weighted_f1": float(f1_score(trues, preds, average="weighted")),
    "matrix": cm.tolist(), "labels": LABELS, "original_reported_accuracy": ORIGINAL_ACCURACY, "comparison": comparison,
    "inference_seconds": EVAL_SECONDS, "model_load_seconds": LOAD_SECONDS, "crops_per_second": len(todo) / EVAL_SECONDS,
    "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "peak_gpu_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    "gpu_status_log": gpu_log, "host": socket.gethostname(),
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__},
    "started_utc": STARTED, "finished_utc": datetime.datetime.utcnow().isoformat() + "Z",
}
with open(f"{OUT_DIR}/results_{SUFFIX}.json", "w") as fh:
    json.dump(result, fh, indent=1)
pd.DataFrame({"test_index": test_df["test_index"], "image_path": test_df["image_path"], "true_label": [LABELS[t] for t in trues], "pred_label": [LABELS[p] for p in preds],
              "confidence": conf, "source": test_df["source"]}).to_csv(f"{OUT_DIR}/predictions_{SUFFIX}.csv", index=False)
if os.path.exists(CKPT):
    os.remove(CKPT)
print(f"Saved: {OUT_DIR}/results_{SUFFIX}.json and predictions_{SUFFIX}.csv")
dbutils.notebook.exit(json.dumps({k: result[k] for k in ("key", "preprocessing", "n", "accuracy", "comparison", "crops_per_second", "peak_gpu_memory_allocated_gb")}))
