# 📰 Indian News Bias Classifier - Project Explanation

## 🎯 Project Goal

The goal of this project was to **classify the political bias (Left / Center / Right)** of Indian news articles.
It covers the **entire AI data pipeline** — from data collection to model training and inference — all built from scratch using real Indian media sources.

---

## 🧠 Motivation

In the Indian media landscape, different outlets often display varying political leanings — sometimes subtly, sometimes overtly.
Manually evaluating these biases is time-consuming and subjective.
So, this project automates the process using **large-scale data extraction**, **LLM-assisted labeling**, and **transformer fine-tuning** to detect ideological inclination in a consistent, data-driven way.

---

| Step                    |                             Purpose                                      |
| ----------------------- | -------------------------------------------------------------------------|
| **1️⃣ Data Extraction** | Query political articles from the **GDELT BigQuery dataset**             |
| **2️⃣ Scraping**        | Use a **multithreaded scraper** to extract full article text             |
| **3️⃣ Labeling**        | Use **Gemini 2.0 Flash API** to assign Left/Center/Right bias            |
| **4️⃣ Model Training**  | Fine-tune **BigBird-RoBERTa** with **LoRA** for long-text classification |
| **5️⃣ Inference**       | Predict bias for new unseen articles                                     |

---

## 🧩 Step 1: Collecting News Data (BigQuery + GDELT)

* Used Google’s **GDELT (Global Database of Events, Language and Tone)** public dataset hosted on **BigQuery**.
* Queried over **100,000+ Indian news article links** from **top 10 Indian outlets** (NDTV, TOI, The Hindu, Republic, etc.)
* Filtered for **politics, elections, economy, government, protests** themes.
* The SQL query randomly sampled 2,000 articles per outlet, ensuring balanced source representation and recent data coverage (2022–2024).

---

## ⚡ Step 2: Scraping Articles (Multithreading + Batch Processing)

This was one of the most critical and optimized parts of your project.

### 🔹 Key Challenges:

* Fetching full article text from 1L+ URLs is time-consuming.
* Some sites throttle requests or have inconsistent HTML structures.
* Need to handle failures gracefully and retry failed URLs.

### 🔹 Your Solution:

* Built a **custom scraping framework** using Python’s `ThreadPoolExecutor`, `queue.Queue`, and the `newspaper3k` library.
* Each thread was assigned a separate `requests.Session()` with retry logic (via `urllib3.Retry`) to avoid connection drops.
* The scraper was **fully multithreaded**, supporting:

  * Parallel processing of **multiple CSVs at once** (each containing thousands of URLs).
  * Thread-safe **logging**, **CSV writing**, and **error recording**.
* Used **batch writing** to CSV (`save_batch()`) every 50 articles to avoid memory overload.
* Implemented random sleep intervals and user-agent rotation to respect rate limits.

✅ Result:

> The scraper efficiently processed **tens of thousands of URLs** concurrently — completing what would have taken hours in **a fraction of the time** while maintaining robust error recovery.

---

## 🤖 Step 3: Labeling Bias using Gemini 2.0 Flash

Once the raw articles were scraped, each article’s text was sent to **Google’s Gemini LLM** for bias annotation.

### 🔹 Key Idea:

Instead of labeling manually, you used the LLM to assign **probabilistic bias scores** based on the article’s tone and framing.

### 🔹 Optimization:

You didn’t send articles one-by-one.
Instead, you **batched 10 articles per API call**, significantly improving throughput and cost-efficiency.

### 🔹 Implementation Details:

* Used **asynchronous requests** via `aiohttp` and `asyncio`.
* Maintained **multiple Gemini API keys** in rotation for parallelism and rate-limit avoidance.
* Implemented **dynamic key selection & cooldown tracking**:
  Each key could send ~12 requests/minute, and the code automatically waited or switched keys when approaching rate limits.
* For each batch, Gemini returned a **JSON array** like:

  ```json
  [
    {"article_id": 0, "bias_scores": {"left": 0.1, "center": 0.8, "right": 0.1}, "explanation": "Neutral reporting"},
    {"article_id": 1, "bias_scores": {"left": 0.7, "center": 0.2, "right": 0.1}, "explanation": "Critical of BJP policies"}
  ]
  ```
* The system validated and normalized the output (ensuring bias scores summed to 1).

✅ Result:

> Efficiently labeled **tens of thousands of news articles** with structured, probabilistic bias scores — all automatically.

---

## 🧠 Step 4: Fine-Tuning the Model (BigBird + LoRA)

### 🔹 Why BigBird?

* Indian news articles are **long (500–1500 tokens)**.
* Standard transformers (like BERT) can’t handle such sequences well.
* **Google’s BigBird-RoBERTa** supports sequences up to **4096 tokens**, making it ideal for long-form political text.

### 🔹 Training Setup:

* Used **LoRA (Low-Rank Adaptation)** to fine-tune only a small subset of parameters —
  drastically reducing memory and training cost while maintaining accuracy.
* Frameworks used:

  * 🤗 `Transformers`
  * `PEFT` (for LoRA)
  * `Evaluate` (for F1 metrics)
* Trained on GPU (mixed precision, FP16), with stratified splits.
* Metrics tracked:

  * **Accuracy**
  * **Macro F1**
  * **Per-class F1 (Left / Center / Right)**

✅ Result:

> Achieved around **82% accuracy** and **~0.79 macro-F1**, demonstrating strong generalization to diverse Indian political language.

---

## 🔍 Step 5: Inference (Prediction)

You built an **inference script** (`step5_infer.py`) that:

* Loads the fine-tuned model and tokenizer.
* Accepts:

  * A **single text input** or
  * A **CSV file** of multiple articles.
* Returns:

  * The predicted bias (`left`, `center`, `right`)
  * The **confidence scores** for each category.

Example:

```bash
python src/step5_infer.py \
  --model_dir models/final \
  --input "Government introduces new subsidy for farmers."
```

**Output:**

```
Predicted Bias: right (L=0.18, C=0.33, R=0.49)
```

✅ Result:

> A ready-to-use CLI tool for classifying any article’s political bias with confidence scores.

---

## 🧰 Tech Stack Summary

| Layer                  | Tools Used                                                        |
| ---------------------- | ----------------------------------------------------------------- |
| **Data Source**        | Google BigQuery (GDELT Dataset)                                   |
| **Scraping**           | Python, Newspaper3k, Requests, ThreadPoolExecutor, Logging        |
| **Annotation**         | Gemini 2.0 Flash API, Aiohttp, Asyncio, Multi-key rate management |
| **Model**              | Google BigBird-RoBERTa (LoRA fine-tuned via HuggingFace)          |
| **Training Framework** | PyTorch, HuggingFace Transformers + PEFT                          |
| **Storage**            | CSV, Parquet                                                      |
| **Automation**         | Shell (`run.sh`), Configs via `configs.yaml`                      |

---

## ⚙️ Key Engineering Highlights

### ✅ Multithreaded Scraping

* Implemented custom **thread-safe scraping** system with locks and queue-based session reuse.
* Enabled high-speed extraction with robust fault tolerance.

### ✅ Batched LLM Labeling

* Handled **multiple articles per API request**, reducing token usage and speeding up labeling.
* Used **async processing + rotating API keys** to maintain concurrency within rate limits.

### ✅ Efficient Fine-Tuning

* Applied **parameter-efficient LoRA** to adapt a large transformer to a niche Indian dataset using limited compute.

### ✅ Fully Modular & Reproducible

* Configurable pipeline (`configs.yaml`) and single command execution (`run.sh`).
* Clear separation of data, code, and model checkpoints.

---

## 📊 Outcomes

* Built a **reproducible pipeline** for political bias detection in Indian news.
* Demonstrated **scalable scraping, LLM labeling, and fine-tuning** on real-world data.
* Produced an interpretable model that provides **confidence scores** for media bias classification.

---

