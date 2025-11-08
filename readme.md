# 📰 Indian News Bias Classifier

> Detect whether Indian news articles lean **Left**, **Center**, or **Right**, using LLM-assisted labeling and transformer fine-tuning.

---

## 🚀 Overview

This project builds an **end-to-end pipeline** that automatically classifies the political bias of Indian news media.
It combines:

* 🗞 **Data Extraction:** GDELT public dataset (Google BigQuery)
* 🕸 **Scraping:** Collect full article text from major Indian news outlets
* 🤖 **LLM Labeling:** Use **Gemini 2.0 Flash** to assign Left/Center/Right bias
* 🧠 **Model Training:** Fine-tune **BigBird-RoBERTa** with LoRA
* 🧾 **Inference:** Predict bias for any new article or dataset

---

## 🧩 Project Pipeline

| Step | Script              | Purpose                             | Output                         |
| ---- | ------------------- | ----------------------------------- | ------------------------------ |
| 1️⃣  | `step1_links.py`    | Query political articles from GDELT | `data/raw/*.csv`               |
| 2️⃣  | `step2_scrape.py`   | Scrape article text and metadata    | `data/interim/scraped.csv`     |
| 3️⃣  | `step3_annotate.py` | Label bias via Gemini API           | `data/processed/annotated.csv` |
| 4️⃣  | `step4_train.py`    | Fine-tune BigBird (LoRA)            | `models/final/`                |
| 5️⃣  | `step5_infer.py`    | Predict bias for new text or CSV    | `data/predictions.csv`         |

---

## 📁 Folder Structure

```
news-bias-india/
├─ configs.yaml
├─ sql/gdelt.sql
├─ src/
│  ├─ step1_links.py
│  ├─ step2_scrape.py
│  ├─ step3_annotate.py
│  ├─ step4_train.py
│  └─ step5_infer.py
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ models/final/
├─ logs/
└─ run.sh
```

---

## ⚙️ Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Add environment variables

Copy `.env.example` → `.env` and fill your keys:

```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-credentials.json
GEMINI_KEY_1=your_gemini_key
GEMINI_KEY_2=your_gemini_key
```

### 3️⃣ Run the full pipeline

```bash
bash run.sh
```

Or manually:

```bash
python src/step1_links.py --config configs.yaml
python src/step2_scrape.py --config configs.yaml
python src/step3_annotate.py --config configs.yaml
python src/step4_train.py --config configs.yaml
```

---

## 🔮 Inference

Predict bias for any text:

```bash
python src/step5_infer.py \
  --model_dir models/final \
  --input "Opposition criticized new tax reforms by the government."
```

**Output:**

```
Predicted Bias: left (L=0.67, C=0.25, R=0.08)
```

Or for a batch CSV:

```bash
python src/step5_infer.py \
  --model_dir models/final \
  --input data/test_articles.csv \
  --output data/predictions.csv
```

---

## 🧠 Model Summary

| Model           | Base                        | Method           | Task                   | Metric (F1-macro) | Metric (Accuracy)
| --------------- | --------------------------- | ---------------- | ---------------------- | ----------------- | --------- |
| BigBird-RoBERTa | google/bigbird-roberta-base | LoRA fine-tuning | 3-class bias detection | ~0.79             | ~82%


---

## ⚖️ Notes

* Uses **GDELT** for large-scale, multilingual political coverage.
* Labels are **LLM-generated** → may include noise or subjectivity.
* Intended for **research / media-analysis**, not for editorial judgment.

---

## 📌 Future Ideas

* Add Hindi/Regional support
* Build simple Streamlit demo for live testing
* Bias-drift visualization over time

---

**Author:** Sidhanth Mandal
🌐 [LinkedIn](https://www.linkedin.com/in/sidhanth-mandal/)