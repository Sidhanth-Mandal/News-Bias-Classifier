import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
import torch
import evaluate

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# CONFIG - change these
MODEL_NAME = "google/bigbird-roberta-base"   
NUM_LABELS = 3
CSV_PATH = "/kaggle/input/news-bias/train.csv"
TEXT_COL = "text"
LABEL_COLS = ["left","center","right"] 
OUTPUT_DIR = "/kaggle/working/outputs_lora"
PER_DEVICE_BATCH = 2                # small due to 16GB
GRAD_ACCUM_STEPS = 8                # effective batch = 16
EPOCHS = 7
LR = 2e-4                           # LoRA-friendly LR
WEIGHT_DECAY = 0.01
WARMUP_PCT = 0.06                   # 6% warmup

# Load csv and build dataset
df = pd.read_csv(CSV_PATH)

def row_to_label(r):
    vals = [r[c] for c in LABEL_COLS]
    return int(np.argmax(vals))  # 0:left,1:center,2:right

df["label"] = df.apply(row_to_label, axis=1)
df = df[[TEXT_COL, "label"]].dropna().reset_index(drop=True)

# Train/val split (stratified)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.12, stratify=df["label"], random_state=SEED)
ds = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df)
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

max_length = 1024 if "bigbird" in MODEL_NAME or "longformer" in MODEL_NAME else 512

def preprocess(ex):
    tok = tokenizer(ex[TEXT_COL], truncation=True, max_length=max_length)
    tok["labels"] = ex["label"]
    return tok

ds = ds.map(preprocess, remove_columns=ds["train"].column_names, batched=False)

# Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
# Prepare for LoRA via PEFT
peft_config = LoraConfig(
    task_type="CAUSAL_LM" if "gpt" in MODEL_NAME else "SEQ_CLS",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["query", "key", "value"] 
)
model = get_peft_model(model, peft_config)


data_collator = DataCollatorWithPadding(tokenizer)

# Metrics
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_macro = metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    # per-class F1
    per = evaluate.load("f1")
    per_class = {f"f1_class_{i}": per.compute(predictions=preds, references=labels, average=None)["f1"][i] for i in range(NUM_LABELS)}
    acc = (preds == labels).mean()
    res = {"f1_macro": f1_macro, "accuracy": acc}
    res.update(per_class)
    return res

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    fp16=True,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
